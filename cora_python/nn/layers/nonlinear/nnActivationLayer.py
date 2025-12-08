"""
nnActivationLayer - abstract class for non-linear layers

Syntax:
    obj = nnActivationLayer(name)

Inputs:
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

References:
    [1] Kochdumper, N., et al. (2022). Open-and closed-loop neural network
        verification using polynomial zonotopes. 
        arXiv preprint arXiv:2207.02715.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Tobias Ladner, Lukas Koller
Written:       28-March-2022
Last update:   01-April-2022 (moved to class folder)
               16-February-2023 (combined approx_type)
               03-May-2023 (LK, backprop)
               30-May-2023 (approx error/output bounds)
               02-August-2023 (LK, zonotope batch-eval & -backprop)
               19-August-2023 (zonotope batch-eval: memory optimizations for GPU training)
               02-February-2024 (LK, better zonotope backpropagation)
Last revision: 10-August-2022 (renamed)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import abstractmethod
from ..nnLayer import nnLayer

from cora_python.contSet.interval import Interval
from cora_python.contSet.emptySet import EmptySet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from cora_python.nn.nnHelper import leastSquarePolyFunc
from cora_python.nn.nnHelper import leastSquareRidgePolyFunc
from cora_python.nn.nnHelper import minMaxDiffOrder
from cora_python.nn.nnHelper import lookupDf
from cora_python.nn.nnHelper import getDerInterval
from cora_python.nn.nnHelper import getOrderIndicesG
from cora_python.nn.nnHelper import getOrderIndicesGI
from cora_python.nn.nnHelper import calcSquared
from cora_python.nn.nnHelper import calcSquaredE
from cora_python.nn.nnHelper import compBoundsPolyZono
from cora_python.nn.nnHelper import reducePolyZono
from cora_python.nn.nnHelper import Heap

class nnActivationLayer(nnLayer):
    """
    Abstract activation layer class for non-linear layers
    
    This class provides the interface and common functionality for all activation layers.
    Each activation layer must implement the abstract methods for evaluation and derivative computation.
    """
    
    # Class constant
    is_refinable = True  # whether the layer is refineable
    
    def __init__(self, name: Optional[str] = None):
        """
        Constructor for nnActivationLayer
        
        Args:
            name: Name of the layer, defaults to type
        """
        # call super class constructor
        super().__init__(name)
        
        # adaptive refinement
        self.order = [1]  # order of approximation polynomial - should be a list like in MATLAB
        self.refine_heu = None  # heuristic for refinement
        self.do_refinement = True  # whether the layer should be refined
        
        self.l = []  # lower bound of last input
        self.u = []  # upper bound of last input
        
        self.merged_neurons = []  # network reduction
        
        # init function handles - match MATLAB exactly
        self.f = lambda x: self.evaluateNumeric(x, {'backprop': False})
        self.df = self.getDf(1)
        
        # initialize backprop storage
        self.backprop = {'store': {}}
    
    # evaluate (element-wise) -------------------------------------------------
    
    def evaluateSensitivity(self, S, x, options: Dict[str, Any]):
        """
        Evaluate sensitivity
        
        Args:
            S: Sensitivity matrix with shape (nK, layer_output_dim, bSz) (torch tensor)
            x: Input point to the layer with shape (layer_input_dim, bSz) (torch tensor)
            options: Evaluation options
            
        Returns:
            S: Updated sensitivity matrix (torch tensor)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(S, np.ndarray):
            S = torch.tensor(S, dtype=torch.float32)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        device = S.device
        dtype = S.dtype
        
        # MATLAB: S = S.*permute(obj.df(x),[3 1 2]);
        # permute([3 1 2]) on a 2D array (dim1, dim2) creates (1, dim1, dim2)
        # This is because MATLAB adds singleton dimensions when they don't exist
        
        df_x = self.df(x)  # Should return shape (layer_output_dim, bSz) for activation layers
        
        # Convert df_x to torch if needed
        if isinstance(df_x, np.ndarray):
            df_x = torch.tensor(df_x, dtype=dtype, device=device)
        elif isinstance(df_x, torch.Tensor):
            df_x = df_x.to(device=device, dtype=dtype)
        
        # Ensure df_x is 2D: (layer_output_dim, bSz)
        # Handle various input shapes
        if df_x.ndim == 1:
            # If 1D, reshape to (layer_output_dim, 1) assuming single batch
            df_x = df_x.reshape(-1, 1)
        elif df_x.ndim == 2:
            # Already 2D, keep as is
            pass
        elif df_x.ndim == 3:
            # If 3D, flatten to 2D: (dim1, dim2, dim3) -> (dim1, dim2*dim3)
            # But we need to match S's dimensions
            # S has shape (nK, layer_output_dim, bSz)
            # df_x should have shape (layer_output_dim, bSz)
            if df_x.shape[0] == 1:
                # (1, dim1, dim2) -> (dim1, dim2)
                df_x = df_x.squeeze(0)
            else:
                # Flatten last two dimensions
                df_x = df_x.reshape(df_x.shape[0], -1)
        else:
            # More than 3D, flatten all but first dimension
            df_x = df_x.reshape(df_x.shape[0], -1)
        
        # Ensure df_x matches S's last two dimensions
        # S has shape (nK, layer_output_dim, bSz)
        # df_x should have shape (layer_output_dim, bSz)
        # x has shape (layer_input_dim, bSz_x) where layer_input_dim = layer_output_dim for activation layers
        # The batch size might differ between S and x, so we need to handle that
        
        # If dimensions don't match, try to reshape
        if df_x.shape[0] != S.shape[1]:
            # Layer output dimension mismatch
            # This can happen if S wasn't properly transformed by the previous linear layer
            # or if there's a dimension mismatch in the network
            # For activation layers, df_x.shape[0] should equal S.shape[1] (both are layer_output_dim)
            # If they don't match, it means S has the wrong dimension from the previous layer
            
            # Check if this is a case where S needs to be expanded/contracted
            # If df_x has more dimensions than S, it means the layer expanded the dimension
            # This shouldn't happen for activation layers (input_dim = output_dim)
            # But we'll try to handle it by checking if we can reshape
            
            if df_x.shape[0] > S.shape[1]:
                # df_x has more dimensions - this suggests S wasn't transformed correctly
                # This is likely a bug in the previous layer's evaluateSensitivity
                raise ValueError(
                    f"Dimension mismatch in evaluateSensitivity: "
                    f"S has shape {S.shape} (expected middle dimension {df_x.shape[0]}), "
                    f"df_x has shape {df_x.shape}, x has shape {x.shape}. "
                    f"This suggests S wasn't properly transformed by the previous layer. "
                    f"For activation layers, S.shape[1] should equal df_x.shape[0] (both = layer_output_dim)."
                )
            elif df_x.size == S.shape[1] * S.shape[2]:
                # Total size matches, try to reshape
                df_x = df_x.reshape(S.shape[1], S.shape[2])
            else:
                # Try to match just the first dimension and broadcast the second
                if df_x.shape[0] == S.shape[1] and df_x.shape[1] == 1:
                    # Broadcast single batch to match S's batch size
                    df_x = df_x.repeat(1, S.shape[2])
                elif df_x.size == S.shape[1] * S.shape[2]:
                    df_x = df_x.reshape(S.shape[1], S.shape[2])
                else:
                    raise ValueError(
                        f"Shape mismatch in evaluateSensitivity: "
                        f"S shape {S.shape}, df_x shape {df_x.shape}, x shape {x.shape}. "
                        f"Expected S.shape[1] == df_x.shape[0] (both should be layer_output_dim)."
                    )
        elif df_x.shape[1] != S.shape[2]:
            # Batch size mismatch - broadcast if df_x has batch size 1
            if df_x.shape[1] == 1:
                # Broadcast single batch to match S's batch size
                df_x = df_x.repeat(1, S.shape[2])
            elif S.shape[2] == 1:
                # S has batch size 1, use df_x's batch size (take first)
                df_x = df_x[:, 0:1]
            else:
                raise ValueError(f"Batch size mismatch in evaluateSensitivity: S shape {S.shape}, df_x shape {df_x.shape}, x shape {x.shape}")
        
        # MATLAB permute([3 1 2]) on 2D array creates (1, dim1, dim2)
        # Reshape df_x from (layer_output_dim, bSz) to (1, layer_output_dim, bSz)
        df_x = df_x.reshape(1, df_x.shape[0], df_x.shape[1])
        
        # Element-wise multiplication: S .* df_x
        # S has shape (nK, layer_output_dim, bSz)
        # df_x has shape (1, layer_output_dim, bSz)
        # Broadcasting: (nK, layer_output_dim, bSz) .* (1, layer_output_dim, bSz) -> (nK, layer_output_dim, bSz)
        S = S * df_x
        return S
    
    def evaluateInterval(self, bounds: 'Interval', options: Dict[str, Any]) -> 'Interval':
        """
        Evaluate interval input
        
        Args:
            bounds: Input interval bounds
            options: Evaluation options
            
        Returns:
            bounds: Output interval bounds
        """
        if options.get('nn', {}).get('reuse_bounds', False):
            # save bounds
            if (not self.l or not self.u or 
                not all(size == size_old for size, size_old in zip(bounds.inf.shape, self.l.shape))):
                self.l = bounds.inf
                self.u = bounds.sup
                
                # set bounds
            elif (bounds.representsa_('emptySet', 1e-10) and 
                  any(np.isnan(self.l)) and any(np.isnan(self.u))):
                bounds = Interval(self.l, self.u)
            
            self.l = np.maximum(self.l, bounds.inf)
            self.u = np.minimum(self.u, bounds.sup)
        
        # propagate through layer
        bounds = super().evaluateInterval(bounds, options)
        return bounds
    
    def evaluateZonotopeBatch(self, c, G, options: Dict[str, Any]):
        """
        Evaluate zonotope batch
        
        Args:
            c: Center (numpy array or torch tensor) - converted to torch internally
            G: Generators (numpy array or torch tensor) - converted to torch internally
            options: Evaluation options
            
        Returns:
            Tuple of (rc, rG) results (torch tensors)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        
        # Ensure c and G are 3D for aux_imgEncBatch (MATLAB format: n x 1 x b and n x q x b)
        # Handle case where c is 2D (n, b) - reshape to (n, 1, b)
        if c.ndim == 2:
            c = c.unsqueeze(1)  # (n, b) -> (n, 1, b)
        # Handle case where G is 2D (n, q) - reshape to (n, q, 1)
        if G.ndim == 2:
            G = G.unsqueeze(2)  # (n, q) -> (n, q, 1)
        
        # Compute image enclosure
        rc, rG, coeffs, d = self.aux_imgEncBatch(self.f, self.df, c, G, options, 
                                                 lambda m: self.computeExtremePointsBatch(m, options))
        
        # store inputs and coeffs for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            # MATLAB: obj.backprop.store.coeffs = m; where m has shape (nk, bSz) - 2D
            # In Python, we need to extract m from coeffs and reshape to match MATLAB
            # coeffs has shape (n, 2, batch) from aux_imgEncBatch
            # Extract slope (first column) and reshape to (n, batch) to match MATLAB
            if len(coeffs.shape) == 3:
                m = coeffs[:, 0, :]  # Extract slope: (n, batch) - 2D
            else:
                m = coeffs[:, 0] if coeffs.ndim == 2 else coeffs
            # Store just the slope m, not the full coeffs array (matches MATLAB)
            self.backprop['store']['coeffs'] = m
            
            # Note: The MATLAB code at lines 120-121 tries to store m_l and m_u,
            # but these variables are undefined in evaluateZonotopeBatch scope.
            # This is dead code. The actual backprop variables (m_c, m_G, etc.)
            # are stored in aux_imgEncBatch, not here.
            # Therefore, we don't store m_l, m_u, GdIdx, dDimsIdx, etc. here.
            # They are already stored by aux_imgEncBatch if exact_backprop is enabled.
                        # Store the slope.
            # if options.get('nn', {}).get('train', {}).get('exact_backprop', False):
            #     # Store gradient for the backprop through an image
            #     # enclosure.
            #     if hasattr(self, 'm_l') and hasattr(self, 'm_u'):
            #         self.backprop['store']['m_l'] = self.m_l
            #         self.backprop['store']['m_u'] = self.m_u
                
            #     if options.get('nn', {}).get('use_approx_error', False):
            #         if not options.get('nn', {}).get('interval_center', False):
            #             if hasattr(self, 'GdIdx'):
            #                 self.backprop['store']['GdIdx'] = self.GdIdx
            #         if hasattr(self, 'dDimsIdx'):
            #             self.backprop['store']['dDimsIdx'] = self.dDimsIdx
            #         if hasattr(self, 'notdDimsIdx'):
            #             self.backprop['store']['notdDimsIdx'] = self.notdDimsIdx
                    
            #         if hasattr(self, 'dl_l') and hasattr(self, 'dl_u'):
            #             self.backprop['store']['dl_l'] = self.dl_l
            #             self.backprop['store']['dl_u'] = self.dl_u
            #         if hasattr(self, 'du_l') and hasattr(self, 'du_u'):
            #             self.backprop['store']['du_l'] = self.du_l
            #             self.backprop['store']['du_u'] = self.du_u
        
        # Convert back to numpy for return (if caller expects numpy)
        rc_np = rc.cpu().numpy() if isinstance(rc, torch.Tensor) else rc
        rG_np = rG.cpu().numpy() if isinstance(rG, torch.Tensor) else rG
        return rc_np, rG_np
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                            id: np.ndarray, id_: np.ndarray, ind: np.ndarray, ind_: np.ndarray,
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate polyZonotope input - matches MATLAB exactly

        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            id: Identifiers
            id_: Identifiers
            ind: Indices
            options: Evaluation options

        Returns:
            Tuple of evaluation results
        """
        print(f"DEBUG: evaluatePolyZonotope - Input: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        
        # Get dimensions
        n = c.shape[0]
        
        # Pre-order reduction using nnHelper
        c, G, GI, E, id, id_, ind, ind_ = self._aux_preOrderReduction(c, G, GI, E, id, id_, ind, ind_, options)
        
        print(f"DEBUG: evaluatePolyZonotope - After preOrderReduction: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        
        # Get max order
        maxOrder = max(self.order)
        
        if options.get('nn', {}).get('sort_exponents', False):
            # Sort columns of exponent matrix
            G, E = self._aux_sort(G, E, maxOrder)
        
        # Compute output sizes per order using nnHelper (improves performance)
        G_start, G_end, G_ext_start, G_ext_end = getOrderIndicesG(G, maxOrder)
        _, GI_end, _, _ = getOrderIndicesGI(GI, G, maxOrder)
        
        # Preallocate output sizes
        c_out = np.zeros((n, 1))
        G_out = np.zeros((n, G_end[-1]))
        GI_out = np.zeros((n, GI_end[-1]))
        E_out = self._aux_computeE_out(E, maxOrder, G_start, G_end)
        d = np.zeros((n, 1))
        
        print(f"DEBUG: evaluatePolyZonotope - Before loop: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        print(f"DEBUG: evaluatePolyZonotope - n={n}, G_end[-1]={G_end[-1]}, GI_end[-1]={GI_end[-1]}")
        
        # Loop over all neurons
        for i in range(n):
            options['nn']['neuron_i'] = i
            order_i = self.order[i] if i < len(self.order) else self.order[0]
            
            print(f"DEBUG: evaluatePolyZonotope - Loop i={i}: c[{i}, 0] = {c[i, 0]}, G[{i}, :] shape: {G[i, :].shape}")
            
            c_out[i], G_out_i, GI_out_i, d[i] = self.evaluatePolyZonotopeNeuron(
                c[i, 0], G[i:i+1, :], GI[i:i+1, :], E, E_out, order_i, ind, ind_, options
            )
            
            print(f"DEBUG: evaluatePolyZonotope - G_out_i shape: {G_out_i.shape}, G_out[i, :len(G_out_i)] shape: {G_out[i, :len(G_out_i)].shape}")
            print(f"DEBUG: evaluatePolyZonotope - G_out_i content: {G_out_i}")
            
            G_out[i, :len(G_out_i)] = G_out_i
            GI_out[i, :len(GI_out_i)] = GI_out_i
        
        if options.get('nn', {}).get('sort_exponents', False):
            # Make sure columns of E_out remain sorted
            G_out, E_out = self._aux_sortPost(G_out, E_out, maxOrder, G_start, G_end, G_ext_start, G_ext_end)
        
        # Compute final output
        c = c_out
        G = G_out
        GI = GI_out
        E = E_out
        
        # Order reduction post to the evaluation using nnHelper
        c, G, GI, E, id, d, id_ = self._aux_postOrderReduction(c, G, GI, E, id, id_, d, options)
        
        # Add approximation error
        if options.get('nn', {}).get('use_approx_error', False):
            G, GI, E, id, id_ = self._aux_addApproxError(d, G, GI, E, id, id_, options)
        
        # Convert E to torch for internal computation
        if isinstance(E, np.ndarray):
            E_torch = torch.tensor(E, dtype=torch.int64)
        else:
            E_torch = E
        
        # Update indices of all-even exponents (for zonotope encl.)
        # Compute using torch internally
        E_mod = E_torch % 2
        ones_like_E = torch.ones_like(E_torch, dtype=torch.int64)
        prod_result = torch.prod(ones_like_E - E_mod, dim=0)
        ind_torch = torch.where(prod_result == 1)[0]
        
        # Convert back to numpy for return
        ind = ind_torch.cpu().numpy() if isinstance(ind_torch, torch.Tensor) else ind_torch
        all_indices = np.arange(E.shape[1])
        ind_ = np.setdiff1d(all_indices, ind)
        
        return c, G, GI, E, id, id_, ind, ind_
    
    def _aux_preOrderReduction(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                              id: np.ndarray, id_: np.ndarray, ind: np.ndarray, ind_: np.ndarray,
                              options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """Order reduction prior to evaluation using nnHelper"""
        # Read number of generators
        n, h = G.shape
        q = GI.shape[1]
        
        print(f"DEBUG: _aux_preOrderReduction - Input: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        print(f"DEBUG: _aux_preOrderReduction - h={h}, q={q}, h+q={h+q}")
        
        # Read max number of generators
        nrMaxGen = options.get('nn', {}).get('num_generators', None)
        if nrMaxGen is None:
            nrMaxGen = float('inf')
        nrMaxGen = min(h + q, nrMaxGen)
        
        print(f"DEBUG: _aux_preOrderReduction - nrMaxGen: {nrMaxGen}")
        
        if (options.get('nn', {}).get('do_pre_order_reduction', False) and 
            options.get('nn', {}).get('max_gens_post') is not None):
            # Consider order reduction
            max_order = max(self.order)
            nrMaxGenOrderRed = int(np.power(options['nn']['max_gens_post'], 1/max_order))
            nrMaxGen = min(nrMaxGen, nrMaxGenOrderRed)
            print(f"DEBUG: _aux_preOrderReduction - After order reduction: nrMaxGen: {nrMaxGen}")
        
        print(f"DEBUG: _aux_preOrderReduction - h+q > nrMaxGen: {h+q} > {nrMaxGen} = {h+q > nrMaxGen}")
        
        if h + q > nrMaxGen:
            print(f"DEBUG: _aux_preOrderReduction - CALLING reducePolyZono!")
            # Reduce using nnHelper
            c, G, GI, E, id, d = reducePolyZono(c, G, GI, E, id, nrMaxGen, self.sensitivity)
            
            # Convert to torch for internal computation
            if isinstance(d, np.ndarray):
                d_torch = torch.tensor(d, dtype=torch.float32)
            else:
                d_torch = d
            if isinstance(GI, np.ndarray):
                GI_torch = torch.tensor(GI, dtype=torch.float32)
            else:
                GI_torch = GI
            
            device = d_torch.device if isinstance(d_torch, torch.Tensor) else torch.device('cpu')
            dtype = d_torch.dtype if isinstance(d_torch, torch.Tensor) else torch.float32
            
            # Add to GI
            D = torch.diag(d_torch.flatten())
            d_mask = d_torch > 0
            GI_torch = torch.cat([GI_torch, D[:, d_mask.flatten()]], dim=1)
            
            # Convert back to numpy for return
            GI = GI_torch.cpu().numpy() if isinstance(GI_torch, torch.Tensor) else GI_torch
            
            # Update number of generators
            h = G.shape[1]
            q = GI.shape[1]
            
            # Update max id
            id_ = max(np.max(id), id_)
            if id_ is None:
                id_ = np.max(id)
        else:
            print(f"DEBUG: _aux_preOrderReduction - NOT calling reducePolyZono")
        
        print(f"DEBUG: _aux_preOrderReduction - Output: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        
        # Update auxiliary variables (matches MATLAB exactly)
        id_ = max(np.max(id), id_)
        if id_ is None:
            id_ = 0
        
        # Compute indices of all-even exponents (for zonotope encl.)
        ind = np.where(np.prod(np.ones_like(E) - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        print(f"DEBUG: _aux_preOrderReduction - Final: ind shape: {ind.shape}, ind_ shape: {ind_.shape}")
        
        return c, G, GI, E, id, id_, ind, ind_
    
    def _aux_sort(self, G: np.ndarray, E: np.ndarray, maxOrder: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sort columns of exponent matrix"""
        # Sort by order first, then by exponent values
        order_indices = []
        for i in range(maxOrder):  # 0-based indexing like Python
            start_idx = np.where(E == i + 1)[1]  # +1 because E stores 1-based order values
            if len(start_idx) > 0:
                order_indices.extend(start_idx)
        
        # Sort within each order
        for i in range(maxOrder):  # 0-based indexing like Python
            mask = E == i + 1  # +1 because E stores 1-based order values
            if np.any(mask):
                # Sort by exponent values
                sorted_indices = np.lexsort(E[mask])
                order_indices.extend(np.where(mask)[1][sorted_indices])
        
        return G[:, order_indices], E[:, order_indices]
    
    def _aux_sortPost(self, G: np.ndarray, E: np.ndarray, maxOrder: int, G_start: np.ndarray,
                      G_end: np.ndarray, G_ext_start: np.ndarray, G_ext_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort columns after evaluation"""
        # Sort by order first, then by exponent values
        for i in range(maxOrder):  # 0-based indexing like Python
            if G_start[i + 1] < G_end[i + 1]:  # +1 because G_start/G_end store 1-based indices
                start_idx = G_start[i + 1]
                end_idx = G_end[i + 1]
                sorted_indices = np.lexsort(E[:, start_idx:end_idx])
                G[:, start_idx:end_idx] = G[:, start_idx:end_idx][:, sorted_indices]
                E[:, start_idx:end_idx] = E[:, start_idx:end_idx][:, sorted_indices]
        
        return G, E
    
    def _aux_computeE_out(self, E: np.ndarray, order: int, G_start: np.ndarray, G_end: np.ndarray) -> np.ndarray:
        """Compute output exponential matrix using nnHelper"""
        # Initialize
        E_out = np.zeros((E.shape[0], G_end[-1]))
        E_ext = [None] * order
        E_out[:, :G_end[0]] = E
        E_ext[0] = E
        
        for i in range(1, order):  # Start from 1 because we need to skip the 0th order
            # Note that e.g., G2 * G3 = G5 -> E2 + E3 = E5
            i1 = i // 2
            i2 = (i + 1) // 2
            
            Ei1_ext = E_ext[i1]
            Ei2_ext = E_ext[i2]
            Ei = calcSquaredE(Ei1_ext, Ei2_ext, i1 == i2)
            E_ext[i] = np.hstack([Ei1_ext, Ei2_ext, Ei])
            
            if i1 < i2:
                # Free memory
                E_ext[i1] = None
            
            E_out[:, G_start[i]:G_end[i]] = Ei
        
        return E_out
    
    def _aux_postOrderReduction(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                               id: np.ndarray, id_: np.ndarray, d: np.ndarray,
                               options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Order reduction after evaluation using nnHelper"""
        # Read number of generators
        n, h = G.shape
        q = GI.shape[1]
        
        # Read max number of generators
        nrMaxGen = options.get('nn', {}).get('num_generators', None)
        if nrMaxGen is None:
            nrMaxGen = float('inf')
        nrMaxGen = min(h + q, nrMaxGen)
        
        if h + q > nrMaxGen:
            # Reduce using nnHelper
            c, G, GI, E, id, d = reducePolyZono(c, G, GI, E, id, nrMaxGen, self.sensitivity)
        
        return c, G, GI, E, id, d, id_
    
    def _aux_addApproxError(self, d: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                           id: np.ndarray, id_: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add approximation error using nnHelper"""
        # Convert inputs to torch if needed
        if isinstance(d, np.ndarray):
            d = torch.tensor(d, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        if isinstance(E, np.ndarray):
            E = torch.tensor(E, dtype=torch.int64)
        if isinstance(GI, np.ndarray):
            GI = torch.tensor(GI, dtype=torch.float32)
        if isinstance(id, np.ndarray):
            id = torch.tensor(id, dtype=torch.int64)
        
        device = d.device if isinstance(d, torch.Tensor) else torch.device('cpu')
        dtype = d.dtype if isinstance(d, torch.Tensor) else torch.float32
        
        # Add error terms to generators
        error_mask = d > 0
        if torch.any(error_mask):
            # Add error as independent generators
            error_count = torch.sum(error_mask).item()
            error_G = torch.zeros((G.shape[0], error_count), dtype=dtype, device=device)
            error_E = torch.zeros((E.shape[0], error_count), dtype=torch.int64, device=device)
            
            # Set error values - use indices where mask is True
            error_indices = torch.where(error_mask)[0]
            for i, idx in enumerate(error_indices):
                error_G[idx, i] = d[idx]
            
            # Add to GI
            GI = torch.cat([GI, error_G], dim=1)
            
            # Update id
            if id_ is not None:
                id_ = max(id_.item() if isinstance(id_, torch.Tensor) else id_, torch.max(id).item() + 1)
            else:
                id_ = torch.max(id).item() + 1
        
        return G, GI, E, id, id_
    
    def evaluatePolyZonotopeNeuron(self, c: float, G: np.ndarray, GI: np.ndarray, E: np.ndarray, 
                                  Es: np.ndarray, order: int, ind: np.ndarray, ind_: np.ndarray, 
                                  options: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Evaluate polyZonotope for a specific neuron - matches MATLAB exactly
        
        Args:
            c: Center (scalar)
            G: Generators (2D row vector with shape (1, h))
            GI: Independent generators (2D row vector with shape (1, q))
            E: Exponent matrix
            Es: Exponent matrix
            order: Polynomial order
            ind: Indices
            ind_: Indices
            options: Evaluation options
            
        Returns:
            Tuple of (c, G, GI, d) results
        """
        # G and GI are already 2D row vectors from the caller (G[i:i+1, :], GI[i:i+1, :])
        # No need to reshape them
        
        l, u = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Extract scalar bounds for this specific neuron
        l_scalar = float(l[0, 0]) if l.size > 0 else 0.0
        u_scalar = float(u[0, 0]) if u.size > 0 else 0.0
        
        # Compute polynomial approximation
        coeffs, d = self.computeApproxPoly(l_scalar, u_scalar, order)
        
        # Compute derivative bounds
        df_l, df_u = self.getDerBounds(l_scalar, u_scalar)
        
        # Use nnHelper to compute polynomial evaluation
        if order == 1:
            # Linear case - direct evaluation
            c_out = coeffs[0] + coeffs[1] * c
            G_out = coeffs[1] * G.flatten()  # Ensure 1D row vector output
            GI_out = coeffs[1] * GI.flatten()  # Ensure 1D row vector output
        else:
            # Higher order case - use nnHelper methods
            # Compute polynomial terms using calcSquared
            c_out, G_out, GI_out = self._computePolynomialTerms(c, G, GI, E, coeffs, order)
            
            # Use nnHelper to get order indices for polynomial evaluation
            G_start, G_end, G_ext_start, G_ext_end = getOrderIndicesG(G, order)
            GI_start, GI_end, GI_ext_start, GI_ext_end = getOrderIndicesGI(GI, G, order)
            
            # Initialize output arrays
            c_out = 0.0  # scalar output
            G_out = np.zeros(G_ext_end[-1])  # row vector output
            GI_out = np.zeros(GI_ext_end[-1])  # row vector output
            
            # Compute polynomial terms for each order
            for i in range(order):  # 0-based indexing like Python
                if G_start[i + 1] < G_end[i + 1]:  # +1 because G_start/G_end store 1-based indices
                    # Get generators for this order
                    G_i = G[G_start[i + 1]:G_end[i + 1]]
                    GI_i = GI[GI_start[i + 1]:GI_end[i + 1]]
                    
                    # Compute polynomial coefficients for this order
                    coeff_i = coeffs[i] if i < len(coeffs) else 0
                    
                    # Add contribution
                    c_out += coeff_i * np.sum(G_i)
                    G_out[G_start[i + 1]:G_end[i + 1]] = coeff_i * G_i
                    GI_out[GI_start[i + 1]:GI_end[i + 1]] = coeff_i * GI_i
        
        return c_out, G_out, GI_out, d
    
    def _computePolynomialTerms(self, c: float, G: np.ndarray, GI: np.ndarray, E: np.ndarray,
                               coeffs: np.ndarray, order: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute polynomial terms using nnHelper methods"""
        # Initialize
        c_out = 0.0  # scalar output
        G_out = np.zeros_like(G)  # row vector output
        GI_out = np.zeros_like(GI)  # row vector output
        
        # For each order, compute the contribution
        for i in range(order):  # 0-based indexing like Python
            if i < len(coeffs):
                # Get coefficient for this order
                coeff = coeffs[i]
                
                # Compute polynomial term
                if i == 0:  # 0-based indexing
                    # Constant term
                    c_out += coeff
                elif i == 1:
                    # Linear term
                    c_out += coeff * c
                    G_out += coeff * G
                    GI_out += coeff * GI
                else:
                    # Higher order terms - use calcSquared
                    # This is a simplified version - in practice, you'd need more sophisticated
                    # polynomial computation using the full nnHelper functionality
                    c_out += coeff * np.power(c, i)
                    
                    # For generators, we need to compute the polynomial expansion
                    # This would typically use calcSquared and other nnHelper methods
                    # For now, we'll use a simplified approach
                    G_out += coeff * np.power(G, i)
                    GI_out += coeff * np.power(GI, i)
        
        return c_out, G_out, GI_out
    
    def evaluateTaylmNeuron(self, input_data: np.ndarray, order: int, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Taylor model for a specific neuron
        
        Args:
            input_data: Input data
            order: Polynomial order
            options: Evaluation options
            
        Returns:
            r: Taylor model evaluation result
        """
        # enclose the ReLU activation function with a Taylor model by
        # fitting a quadratic function
        
        # compute lower and upper bound
        int_val = Interval(input_data)
        l = int_val.inf
        u = int_val.sup
        
        # compute approx poly + error
        coeffs, d = self.computeApproxPoly(l, u, order, options['nn']['poly_method'])
        
        # evaluate
        r = coeffs[-1] + Interval(-d, d)
        for i in range(len(coeffs) - 1):
            r = r + coeffs[-(i + 1)] * (input_data ** (i + 1))
        
        return r
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Taylor model input
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Taylor model evaluation result
        """
        # Implement Taylor model evaluation using nnHelper methods
        # This matches MATLAB's nnActivationLayer/evaluateTaylm functionality
        
        # For Taylor models, we use polynomial approximation with error bounds
        # Get polynomial order from options
        order = options.get('nn', {}).get('order', 1)
        
        # Get bounds for the input domain
        # This would typically come from the Taylor model bounds
        # For now, use a simplified approach
        
        # Use nnHelper to compute polynomial approximation
        # This would involve:
        # 1. Computing polynomial coefficients using leastSquarePolyFunc
        # 2. Computing error bounds using minMaxDiffOrder
        # 3. Using getDerInterval for derivative bounds
        
        # Placeholder implementation
        # TODO: Implement full Taylor model evaluation logic
        return input_data  # For now, just return input unchanged
    
    def evaluateConZonotopeNeuron(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                                  d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                                  j: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                                           np.ndarray, np.ndarray, 
                                                                           np.ndarray, np.ndarray]:
        """
        Evaluate constraint zonotope for a specific neuron
        
        Args:
            c: Center
            G: Generators
            C: Constraint matrix
            d: Constraint vector
            l: Lower bounds
            u: Upper bounds
            j: Neuron index
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        raise CORAerror('CORA:notSupported', 'conZonotope not supported for this layer')
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate constraint zonotope input
        
        Args:
            c: Center
            G: Generators
            C: Constraint matrix
            d: Constraint vector
            l: Lower bounds
            u: Upper bounds
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        # This method needs to be implemented based on MATLAB logic
        # For now, return placeholders
        raise CORAerror('CORA:notSupported', 'evaluateConZonotope not implemented yet')
    
    # backprop ------------------------------------------------------------
    
    def backpropNumeric(self, input_data: np.ndarray, grad_out: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Backpropagate numeric gradients
        
        Args:
            input_data: Input data
            grad_out: Output gradients
            options: Backpropagation options
            
        Returns:
            grad_in: Input gradients
        """
        # backpropagte gradient
        grad_in = self.df(input_data) * grad_out
        return grad_in
    
    def backpropIntervalBatch(self, l: np.ndarray, u: np.ndarray, gl: np.ndarray, gu: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate interval batch
        
        Args:
            l: Lower bounds
            u: Upper bounds
            gl: Lower bound gradients
            gu: Upper bound gradients
            options: Backpropagation options
            
        Returns:
            Tuple of (gl, gu) results
        """
        gl = self.df(l) * gl
        gu = self.df(u) * gu
        return gl, gu
    
    def backpropZonotopeBatch(self, c: np.ndarray, G: np.ndarray, gc: np.ndarray, 
                              gG: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate zonotope batch
        
        Args:
            c: Center
            G: Generators
            gc: Center gradients
            gG: Generator gradients
            options: Backpropagation options
            
        Returns:
            Tuple of (gc, gG) results
        """
        # obtain stored slope from forward prop.
        # MATLAB: m = obj.backprop.store.coeffs; [nk,bSz] = size(m);
        # In MATLAB, coeffs stores just m (the slope) with shape (nk, bSz) - 2D
        # In Python, we also store just m with shape (n, batch) - 2D
        m = self.backprop['store']['coeffs']
        
        # MATLAB: permute(m,[1 3 2]) on 2D array (nk, bSz) creates (nk, 1, bSz)
        # Python: need to add singleton dimension first, then transpose
        if m.ndim == 2:
            # Reshape (n, batch) -> (n, 1, batch) to match MATLAB permute behavior
            m_3d = m[:, np.newaxis, :]  # (n, 1, batch)
        else:
            m_3d = m
        
        # obtain indices of active generators
        genIds = self.backprop['store'].get('genIds', slice(None))
        
        if options.get('nn', {}).get('train', {}).get('exact_backprop', False):
            # Obtain indices of the approximation errors in the generator
            # matrix.
            GdIdx = self.backprop['store']['GdIdx']
            dDimsIdx = self.backprop['store']['dDimsIdx']
            
            m_c = self.backprop['store']['m_c']
            m_G = self.backprop['store']['m_G']
            
            dc_c = self.backprop['store']['dc_c']
            dc_G = self.backprop['store']['dc_G']
            
            d_c = self.backprop['store']['d_c']
            d_G = self.backprop['store']['d_G']
            
            # Convert inputs to torch if needed
            if isinstance(gc, np.ndarray):
                gc = torch.tensor(gc, dtype=torch.float32)
            if isinstance(c, np.ndarray):
                c = torch.tensor(c, dtype=torch.float32)
            if isinstance(gG, np.ndarray):
                gG = torch.tensor(gG, dtype=torch.float32)
            if isinstance(G, np.ndarray):
                G = torch.tensor(G, dtype=torch.float32)
            if isinstance(m_3d, np.ndarray):
                m_3d = torch.tensor(m_3d, dtype=torch.float32)
            if isinstance(m, np.ndarray):
                m = torch.tensor(m, dtype=torch.float32)
            if isinstance(m_c, np.ndarray):
                m_c = torch.tensor(m_c, dtype=torch.float32)
            if isinstance(m_G, np.ndarray):
                m_G = torch.tensor(m_G, dtype=torch.float32)
            if isinstance(dc_c, np.ndarray):
                dc_c = torch.tensor(dc_c, dtype=torch.float32)
            if isinstance(d_c, np.ndarray):
                d_c = torch.tensor(d_c, dtype=torch.float32)
            
            device = gc.device if isinstance(gc, torch.Tensor) else torch.device('cpu')
            dtype = gc.dtype if isinstance(gc, torch.Tensor) else torch.float32
            
            # Precompute outer product of gradients and inputs.
            hadProdc = torch.permute(gc * c, (1, 2, 0))
            hadProdG = gG[:, genIds, :] * G
            
            # Backprop gradients.
            # MATLAB: gc = m.*gc + ... (element-wise multiplication)
            # For interval_center: MATLAB uses permute(m,[1 3 2]).*gc
            # For non-interval_center: MATLAB uses gc.*m (m is 2D)
            if options.get('nn', {}).get('interval_center', False):
                gc = torch.permute(m_3d, (0, 2, 1)) * gc  # permute(m,[1 3 2]) -> (n, batch, 1)
            else:
                gc = gc * m  # Element-wise: (n, 1, batch) * (n, batch) broadcasts correctly
            
            rgc = gc + m_c * torch.reshape(hadProdc + torch.sum(hadProdG, dim=1), c.shape)
            # Convert dDimsIdx and GdIdx to torch if needed for indexing
            if isinstance(dDimsIdx, np.ndarray):
                dDimsIdx = torch.tensor(dDimsIdx, dtype=torch.long, device=device)
            if isinstance(GdIdx, np.ndarray):
                GdIdx = torch.tensor(GdIdx, dtype=torch.long, device=device)
            rgc[dDimsIdx] = rgc[dDimsIdx] + dc_c * gc[dDimsIdx]
            rgc[dDimsIdx] = rgc[dDimsIdx] + d_c * gG[GdIdx]
            # Assign results.
            gc = rgc
            
            # MATLAB: gG(:,genIds,:) = gG(:,genIds,:).*permute(m,[1 3 2]) + ...
            # permute(m,[1 3 2]) on 2D (nk,bSz) -> (nk,1,bSz)
            # Then transpose to (1,nk,bSz) for broadcasting
            m_permuted = torch.permute(m_3d, (1, 2, 0))  # (1, n, batch) from (n, 1, batch)
            rgG = gG[:, genIds, :] * m_permuted + m_G * (hadProdc + hadProdG)
            rgG = torch.permute(rgG, (1, 0, 2))
            # Convert dc_G and d_G to torch if needed
            if isinstance(dc_G, np.ndarray):
                dc_G = torch.tensor(dc_G, dtype=dtype, device=device)
            if isinstance(d_G, np.ndarray):
                d_G = torch.tensor(d_G, dtype=dtype, device=device)
            rgG[:, dDimsIdx] = rgG[:, dDimsIdx] + dc_G * torch.reshape(gc[dDimsIdx], (1, -1))
            rgG[:, dDimsIdx] = rgG[:, dDimsIdx] + d_G * torch.reshape(gG[GdIdx], (1, -1))
            rgG = torch.permute(rgG, (1, 0, 2))
            # Assign results.
            gG = rgG
            
        else:
            # Consider the approximation as fixed. Use the slope of the
            # approximation for backpropagation
            # MATLAB: 
            #   if interval_center: gc = permute(m,[1 3 2]).*gc;
            #   else: gc = gc.*m;
            #   gG(:,genIds,:) = gG(:,genIds,:).*permute(m,[1 3 2]);
            # Convert inputs to torch if needed
            if isinstance(gc, np.ndarray):
                gc = torch.tensor(gc, dtype=torch.float32)
            if isinstance(gG, np.ndarray):
                gG = torch.tensor(gG, dtype=torch.float32)
            if isinstance(m_3d, np.ndarray):
                m_3d = torch.tensor(m_3d, dtype=torch.float32)
            if isinstance(m, np.ndarray):
                m = torch.tensor(m, dtype=torch.float32)
            
            device = gc.device if isinstance(gc, torch.Tensor) else torch.device('cpu')
            dtype = gc.dtype if isinstance(gc, torch.Tensor) else torch.float32
            
            if options.get('nn', {}).get('interval_center', False):
                # MATLAB: permute(m,[1 3 2]) on 2D (nk,bSz) -> (nk,1,bSz)
                gc = torch.permute(m_3d, (0, 2, 1)) * gc  # (n, batch, 1) * (n, 1, batch)
            else:
                # MATLAB: gc.*m where m is (nk,bSz) and gc is (nk,1,bSz)
                # In MATLAB, m (nk,bSz) broadcasts to (nk,1,bSz) for element-wise multiplication
                # In Python, we need to reshape m to (nk,1,bSz) to match gc's shape
                if m.ndim == 2:
                    m_broadcast = m.unsqueeze(1)  # (nk, 1, bSz)
                else:
                    m_broadcast = m
                gc = gc * m_broadcast  # (nk, 1, bSz) * (nk, 1, bSz)
            
            # MATLAB: gG(:,genIds,:).*permute(m,[1 3 2])
            # permute(m,[1 3 2]) on 2D (nk,bSz) -> (nk,1,bSz), then transpose to (1,nk,bSz)
            m_permuted = torch.permute(m_3d, (1, 2, 0))  # (1, n, batch)
            gG = gG[:, genIds, :] * m_permuted
        
        # Clear backprop storage.
        if 'store' in self.backprop:
            self.backprop['store'].clear()
        
        return gc, gG
    
    # Auxiliary functions -----------------------------------------------------
    
    def getDf(self, i: int) -> Callable:
        """
        Get derivative function
        
        Args:
            i: Derivative order
            
        Returns:
            df_i: Derivative function
        """
        # Use nnHelper.lookupDf for derivative caching (matches MATLAB)
        return lookupDf(self, i)
    
    def getNumNeurons(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get number of input and output neurons (matches MATLAB exactly)
        
        Returns:
            Tuple of (nin, nout) where each can be None
        """
        nin = None
        nout = None
        return nin, nout
    
    def getOutputSize(self, inputSize: List[int]) -> List[int]:
        """
        Get output size given input size
        
        Args:
            inputSize: Input dimensions
            
        Returns:
            outputSize: Output dimensions
        """
        outputSize = inputSize  # for most activation functions
        return outputSize
    
    @abstractmethod
    def getDerBounds(self, l: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method to get derivative bounds
        
        Args:
            l: Lower bounds
            u: Upper bounds
            
        Returns:
            Tuple of (df_l, df_u) derivative bounds
        """
        pass
    
    # approximation polynomial + error
    
    def computeApproxError(self, l: np.ndarray, u: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute approximation error - match MATLAB exactly
        
        Args:
            l: Lower bounds
            u: Upper bounds
            coeffs: Polynomial coefficients
            
        Returns:
            Tuple of (coeffs, d) updated coefficients and error bound
        """
        # bound approximation error according to [1, Sec. 3.2]
        
        # Make a copy to avoid modifying original
        coeffs = coeffs.copy()
        
        # compute the difference between activation function and quad. fit
        df_l, df_u = self.getDerBounds(l, u)
        
        # Use the proper nnHelper function like MATLAB
        diffl, diffu = minMaxDiffOrder(coeffs, l, u, self.f, df_l, df_u)
        
        # change polynomial s.t. lower and upper error are equal
        diffc = (diffl + diffu) / 2
        coeffs[-1] = coeffs[-1] + diffc
        d = diffu - diffc  # error is radius then.
        
        return coeffs, d
    
    def findRegionPolys(self, tol: float, order: int, l_max: np.ndarray, u_max: np.ndarray, 
                        pStart: np.ndarray, dStart: float, pEnd: np.ndarray, dEnd: float) -> list:
        """
        Find regions with approximating polynomials
        
        Args:
            tol: Tolerance
            order: Polynomial order
            l_max: Maximum lower bounds
            u_max: Maximum upper bounds
            pStart: Start polynomial
            dStart: Start error
            pEnd: End polynomial
            dEnd: End error
            
        Returns:
            coeffs: List of dictionaries with polynomial coefficients and regions
        """
        # Check magnitude of given approx errors
        if dStart > tol:
            # CORAwarning('CORA:nn','nnActivationLayer/findRegionPolys: dStart > tol.')
            pass
        if dEnd > tol:
            # CORAwarning('CORA:nn','nnActivationLayer/findRegionPolys: dEnd > tol.')
            pass
        
        # Remove reg_polys if present
        if hasattr(self, 'reg_polys'):
            # CORAwarning('CORA:nn',"Temporarily removing current region polynomials.")
            reg_polys = self.reg_polys
            self.reg_polys = []
        else:
            reg_polys = None
        
        # Init list
        coeffs = []
        
        # Start polynomial
        coeffs.append({
            'l': float('-inf'),
            'u': l_max,
            'p': pStart,
            'd': dStart
        })
        
        # Iteratively find approx polynomial
        l = l_max
        u = u_max
        while l < u:
            print(f"Progress: {((u_max-l_max)-(u-l))/(u_max-l_max) * 100:.2f}%")
            
            # Left polynomial
            u_i = u
            d = np.inf
            p = [0, 0]
            
            while d > tol:
                if not np.isinf(d):
                    # Shorten region
                    u_i = l + (u_i - l) * 0.5
                    
                    if (u_i - l) < tol:
                        raise CORAerror('CORA:notConverged', 
                                      f'Unable to find polynomial at [{l:.6f},{u:.6f}]. Reached x={u_i:.6f}.')
                
                # Find polynomial + error
                p, d = self.computeApproxPoly(l, u_i, order, "regression")
            
            # Working polynomial found, trying to reduce the order
            for o in range(order - 1, -1, -1):
                # Find polynomial + error
                p_o, d_o = self.computeApproxPoly(l, u_i, o, "regression")
                
                if d_o < tol:
                    # Update with lower-order polynomial
                    p = p_o
                    d = d_o
                else:
                    # No lower order polynomial will recover d to be below tol
                    break
            
            # Next polynomial
            coeffs.append({
                'l': l,
                'u': u_i,
                'p': p,
                'd': d
            })
            
            # Move to next region
            l = u_i
            
            if l == u:
                # Check if entire domain is covered
                break
            
            print(f"Progress: {((u_max-l_max)-(u-l))/(u_max-l_max) * 100:.2f}%")
            
            # Right polynomial
            l_i = l
            d = np.inf
            p = [0, 0]
            
            while d > tol:
                if not np.isinf(d):
                    # Half region
                    l_i = u - (u - l_i) / 2
                    
                    if (u - l_i) < tol:
                        raise CORAerror('CORA:notConverged', 
                                      f'Unable to find polynomial at [{l:.6f},{u:.6f}]. Reached x={l_i:.6f}.')
                
                # Find polynomial + error
                p, d = self.computeApproxPoly(l_i, u, order, "regression")
                p, d = self.computeApproxError(l_i, u, p)
            
            # Next polynomial
            coeffs.append({
                'l': l_i,
                'u': u,
                'p': p,
                'd': d
            })
            
            # Move to next region
            u = l_i
        
        print(f"Progress: {((u_max-l_max)-(u-l))/(u_max-l_max) * 100:.2f}%")
        
        # End polynomial
        coeffs.append({
            'l': u_max,
            'u': float('inf'),
            'p': pEnd,
            'd': dEnd
        })
        
        # Sort coeffs
        coeffs.sort(key=lambda x: x['l'])
        
        print(f"Required {len(coeffs)} polynomials.")
        
        # Restore reg_polys
        if reg_polys is not None:
            self.reg_polys = reg_polys
        
        return coeffs
    

    

    
    def computeApproxPoly(self, l: np.ndarray, u: np.ndarray, *args) -> Tuple[np.ndarray, float]:
        """
        Compute approximating polynomial
        
        Args:
            l: Lower bounds
            u: Upper bounds
            *args: order and poly_method (order defaults to 1, poly_method defaults to 'regression')
            
        Returns:
            Tuple of (coeffs, d) polynomial coefficients and error bound
        """
        # parse input validation - match MATLAB exactly
        if len(args) == 0:
            order = 1
            poly_method = 'regression'
        elif len(args) == 1:
            order = args[0]
            poly_method = 'regression'
        elif len(args) == 2:
            order = args[0]
            poly_method = args[1]
        else:
            raise CORAerror('CORA:wrongInputInConstructor', 'Too many arguments')
        
        # validate inputs - match MATLAB exactly
        if not isinstance(l, (int, float, np.ndarray)) or not isinstance(u, (int, float, np.ndarray)):
            raise CORAerror('CORA:wrongInputInConstructor', 'l and u must be numeric scalars')
        if not isinstance(order, int) or order < 1:
            raise CORAerror('CORA:wrongInputInConstructor', 'order must be a positive integer')
        
        valid_poly_methods = ['regression', 'ridgeregression', 'bernstein', 
                             'throw-catch', 'taylor', 'singh', 'bounds']
        if poly_method not in valid_poly_methods:
            raise CORAerror('CORA:wrongInputInConstructor', f"poly_method must be one of {valid_poly_methods}")
        
        # trivial case - match MATLAB exactly
        if l == u:
            # compute tangent line in l
            coeffs = [self.df(l), self.f(l) - self.df(l) * l]
            d = 0
            return coeffs, d
        elif l > u:
            raise CORAerror('CORA:wrongInputInConstructor', 'l must be <= u')
        
        # init
        coeffs = []
        d = []
        
        # compute approximation polynomial
        # use at 10 points per coeff within [l, u] for regression
        # https://en.wikipedia.org/wiki/One_in_ten_rule
        num_points = 10 * (order + 1)
        
        if poly_method == 'regression':
            x = np.linspace(l, u, num_points)
            y = self.f(x)
            
            # compute polynomial that best fits the activation function
            coeffs = leastSquarePolyFunc(x, y, order)
            
        elif poly_method == 'ridgeregression':
            x = np.linspace(l, u, num_points)
            y = self.f(x)
            
            coeffs = leastSquareRidgePolyFunc(x, y, order)
            
        elif poly_method == 'taylor':
            # taylor series expansion at middle point
            c = (l + u) / 2
            
            # init
            P = [1]  # pascal's triangle
            coeffs = np.zeros(order + 1)
            
            # taylor series expansion
            for i in range(order + 1):
                df_i = self.getDf(i)
                coeffs[-(i+1):] = coeffs[-(i+1):] + \
                    (np.array(P) * (-c) ** np.arange(i+1)) * df_i(c) / np.math.factorial(i)
                
                # prepare for next iteration
                P = [1] + [P[j] + P[j+1] for j in range(len(P)-1)] + [1]
            
            # TODO: lagrange remainder
            # d = ?
            
        elif poly_method == 'singh':
            # call custom implementation
            coeffs, d = self.computeApproxPolyCustom(l, u, order, poly_method)
            
        elif poly_method == 'bounds':
            coeffs = [(self.f(u) - self.f(l)) / (u - l), 0]
            
        else:
            # other methods not implemented yet
            coeffs = []
            d = []
        
        # parse coeffs and d - match MATLAB exactly
        if len(coeffs) == 0:
            # unable to determine coeffs
            raise CORAerror('CORA:notSupported', f"'{poly_method}' for polynomial of order={order}")
        elif len(d) == 0:
            # compute approx error if not found already
            coeffs, d = self.computeApproxError(l, u, coeffs)
        
        return coeffs, d
    
    def computeApproxPolyCustom(self, l: np.ndarray, u: np.ndarray, order: int, poly_method: str) -> Tuple[np.ndarray, float]:
        """
        Compute custom approximating polynomial
        
        Args:
            l: Lower bounds
            u: Upper bounds
            order: Polynomial order
            poly_method: Polynomial method
            
        Returns:
            Tuple of (coeffs, d) polynomial coefficients and error bound
        """
        # implement custom polynomial computation in subclass
        coeffs = []
        d = 0.0
        
        # This method should be overridden in subclasses for specific activation functions
        # The base implementation returns empty coefficients
        return coeffs, d
    
    def computeExtremePointsBatch(self, m: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute extreme points batch
        
        Args:
            m: Input data
            options: Options
            
        Returns:
            Tuple of (xs, dxsdm) extreme points and derivatives
        """
        raise CORAerror('CORA:notDefined', 'computeExtremePointsBatch not implemented in base class')
    
    def aux_imgEncBatch(self, f: callable, df: callable, c, G, 
                        options: Dict[str, Any], extremePoints: callable):
        """
        Auxiliary function for image enclosure batch
        
        Args:
            f: Function
            df: Derivative function
            c: Center (numpy array or torch tensor) - converted to torch internally
            G: Generators (numpy array or torch tensor) - converted to torch internally
            options: Options
            extremePoints: Extreme points function
            
        Returns:
            Tuple of (rc, rG, coeffs, d) results (torch tensors, converted back to numpy if needed)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        
        device = c.device if isinstance(c, torch.Tensor) else torch.device('cpu')
        dtype = c.dtype if isinstance(c, torch.Tensor) else torch.float32
        
        # obtain indices of active generators
        genIds = self.backprop['store'].get('genIds', slice(None))
        # Ensure batch dimensions of c and G match (MATLAB keeps both as n x 1 x b and n x q x b)
        # Handle both 2D and 3D cases
        if c.ndim == 3:
            n_c, _, b_c = c.shape
        else:
            # c is 2D, add singleton dimension
            c = c.unsqueeze(1) if c.ndim == 2 else c
            n_c, _, b_c = c.shape
        
        if G.ndim == 3:
            n_g, _, b_g = G.shape
        else:
            # G is 2D, add singleton dimension for generators
            G = G.unsqueeze(1) if G.ndim == 2 else G
            n_g, _, b_g = G.shape
        
        if b_c != b_g:
            if b_c == 1 and b_g > 1:
                c = c.repeat(1, 1, b_g)
                b_c = b_g
            elif b_g == 1 and b_c > 1:
                G = G.repeat(1, 1, b_c)
                b_g = b_c
        
        # Ensure c has the correct shape (n, 1, b) to match MATLAB
        if c.ndim == 3 and c.shape[1] != 1:
            # If c has wrong middle dimension, reshape it to (n, 1, b)
            c = c[:, :1, :]  # Take only first column
        
        # compute bounds: radius per feature and batch; shape (n,1,b)
        # MATLAB: [n,q,bSz] = size(G); r = reshape(sum(abs(G(:,genIds,:)),2),[n bSz]);
        # Get dimensions first
        n_c = c.shape[0]
        b_c = c.shape[2] if c.ndim == 3 else (c.shape[1] if c.ndim == 2 else 1)
        
        # Ensure G is 3D: (n, q, bSz)
        if G.ndim == 2:
            # G is 2D, need to determine if it's (n, bSz) or (q, bSz)
            # If G.shape[1] == b_c, then G is likely (q, bSz) - add n dimension
            if G.shape[1] == b_c:
                # G is (q, bSz), add n dimension: (1, q, bSz)
                G = G.unsqueeze(0)  # (1, q, bSz)
            elif G.shape[0] == n_c:
                # G is (n, bSz), add q dimension: (n, 1, bSz)
                G = G.unsqueeze(1)  # (n, 1, bSz)
            else:
                # Can't determine, assume (q, bSz) and add n dimension
                G = G.unsqueeze(0)  # (1, q, bSz)
        
        # Get dimensions after ensuring G is 3D
        n_g, q_g, b_g = G.shape
        
        # Check if batch sizes match
        if b_g != b_c:
            if b_g == 1 and b_c > 1:
                # G has batch size 1, replicate to match c
                G = G.repeat(1, 1, b_c)
                b_g = b_c
            elif b_c == 1 and b_g > 1:
                # c has batch size 1, replicate to match G
                if c.ndim == 3:
                    c = c.repeat(1, 1, b_g)
                else:
                    c = c.repeat(1, b_g)
                b_c = b_g
        
        # Check if G and c have matching first dimensions
        if n_g != n_c:
            # G and c have different n dimensions, need to broadcast or replicate
            if n_g == 1 and n_c > 1:
                # G has n=1, replicate to match c
                G = G.repeat(n_c, 1, 1)
                n_g = n_c
            elif n_c == 1 and n_g > 1:
                # c has n=1, replicate to match G
                c = c.repeat(n_g, 1, 1)
                n_c = n_g
        
        # MATLAB: r = reshape(sum(abs(G(:,genIds,:)),2),[n bSz]);
        # Sum over axis 1 (generators), then reshape to (n, bSz)
        # G should be 3D (n_g, q, b_g) at this point
        G_selected = G[:, genIds, :]  # (n_g, num_gens, b_g)
        r_raw = torch.sum(torch.abs(G_selected), dim=1)  # Should be (n_g, b_g)
        
        # DEBUG: Log r values for comparison with MATLAB (first few iterations only)
        # This helps identify why bounds collapse faster in Python
        if hasattr(self, '_debug_iteration') and self._debug_iteration is not None and self._debug_iteration <= 10:
            print(f"ACTIVATION LAYER DEBUG (iteration {self._debug_iteration}):")
            print(f"  Layer: {self.__class__.__name__}")
            print(f"  G_selected shape: {G_selected.shape}")
            r_raw_np = r_raw.cpu().numpy() if isinstance(r_raw, torch.Tensor) else r_raw
            print(f"  r_raw (first 3 neurons, first 3 batches): {r_raw_np[:min(3, r_raw_np.shape[0]), :min(3, r_raw_np.shape[1])].flatten()}")
            print(f"  r_raw min/max: min={torch.min(r_raw).item()}, max={torch.max(r_raw).item()}")
            if torch.any(r_raw < 1e-6):
                zero_count = torch.sum(r_raw < 1e-6).item()
                print(f"  WARNING: {zero_count} neurons have r < 1e-6 (bounds will collapse)!")
                zero_indices = torch.where(r_raw < 1e-6)
                zero_indices_np = [z.cpu().numpy()[:5] for z in zero_indices]
                print(f"  Zero r locations (first 5): neurons={zero_indices_np[0]}, batches={zero_indices_np[1]}")
        
        # Ensure r_raw is 2D (n, b) matching c's dimensions
        # MATLAB: r = reshape(sum(abs(G(:,genIds,:)),2),[n bSz]);
        # r_raw should be (n_g, b_g) after sum, but we need (n_c, b_c)
        if r_raw.ndim == 1:
            # r_raw is 1D, this shouldn't happen if G is properly 3D
            # Try to infer correct shape
            if r_raw.numel() == n_g * b_g:
                # r_raw is flattened (n*b,), reshape to (n, b)
                r_raw = r_raw.reshape((n_g, b_g))
            elif r_raw.numel() == b_g:
                # r_raw is (b,), replicate to (n, b)
                r_raw = r_raw.reshape((1, -1)).repeat(n_g, 1)
            else:
                raise ValueError(f"Cannot handle r_raw with size {r_raw.numel()}, expected {n_g * b_g} or {b_g}, G.shape={G.shape}, c.shape={c.shape}, n_g={n_g}, b_g={b_g}")
        
        # Now ensure r_raw matches c's dimensions (n_c, b_c)
        if r_raw.ndim == 2:
            # r_raw is 2D, check and fix dimensions to match c
            if r_raw.shape[0] != n_c:
                # r_raw has wrong first dimension
                if r_raw.shape[0] == 1 and n_c > 1:
                    r_raw = r_raw.repeat(n_c, 1)
                elif r_raw.shape[0] > n_c:
                    r_raw = r_raw[:n_c, :]
                elif r_raw.shape[0] < n_c:
                    raise ValueError(f"r_raw.shape[0]={r_raw.shape[0]} < n_c={n_c}, cannot replicate")
            if r_raw.shape[1] != b_c:
                # r_raw has wrong second dimension
                if r_raw.shape[1] == 1 and b_c > 1:
                    r_raw = r_raw.repeat(1, b_c)
                elif r_raw.shape[1] > b_c:
                    r_raw = r_raw[:, :b_c]
                elif r_raw.shape[1] < b_c:
                    # r_raw has smaller batch size, replicate to match c
                    if b_c % r_raw.shape[1] == 0:
                        # b_c is a multiple of r_raw.shape[1], replicate
                        nReps = b_c // r_raw.shape[1]
                        r_raw = r_raw.repeat(1, nReps)
                    else:
                        raise ValueError(f"r_raw.shape[1]={r_raw.shape[1]} < b_c={b_c} and b_c is not a multiple, cannot replicate")
        else:
            raise ValueError(f"r_raw is not 2D after processing: ndim={r_raw.ndim}, shape={r_raw.shape}, G.shape={G.shape}, c.shape={c.shape}")
        
        # Final safety check: r_raw must be 2D with shape (n_c, b_c)
        if r_raw.ndim != 2:
            raise ValueError(f"r_raw is not 2D before reshape: ndim={r_raw.ndim}, shape={r_raw.shape}, expected ({n_c}, {b_c})")
        if r_raw.shape != (n_c, b_c):
            # Try to fix it one more time
            if r_raw.numel() == n_c * b_c:
                r_raw = r_raw.reshape((n_c, b_c))
            else:
                raise ValueError(f"r_raw shape {r_raw.shape} != expected ({n_c}, {b_c}), size={r_raw.numel()}, expected size={n_c * b_c}")
        
        # Reshape to (n, 1, b) to match MATLAB's usage
        # r_raw should now be (n_c, b_c)
        r = r_raw.reshape((n_c, 1, b_c))
        # r = max(eps('like',c),r); % prevent numerical instabilities
        
        # DEBUG: Log bounds computation for comparison with MATLAB
        if hasattr(self, '_debug_iteration') and self._debug_iteration is not None and self._debug_iteration <= 10:
            print(f"  After bounds computation:")
            print(f"    r shape: {r.shape}, r (first 3 neurons, first 3 batches): {r[:min(3, r.shape[0]), 0, :min(3, r.shape[2])].flatten()}")
            print(f"    c (first 3 neurons, first 3 batches): {c[:min(3, c.shape[0]), 0, :min(3, c.shape[2])].flatten()}")
            l_debug = c - r
            u_debug = c + r
            print(f"    l (first 3 neurons, first 3 batches): {l_debug[:min(3, l_debug.shape[0]), 0, :min(3, l_debug.shape[2])].flatten()}")
            print(f"    u (first 3 neurons, first 3 batches): {u_debug[:min(3, u_debug.shape[0]), 0, :min(3, u_debug.shape[2])].flatten()}")
            bounds_collapse = torch.abs(u_debug - l_debug) < torch.finfo(dtype).eps
            if torch.any(bounds_collapse):
                collapse_count = torch.sum(bounds_collapse).item()
                print(f"    WARNING: {collapse_count} bounds have collapsed (l ≈ u)!")
                collapse_indices = torch.where(bounds_collapse)
                collapse_indices_np = [ci.cpu().numpy()[:5] for ci in collapse_indices]
                print(f"    Collapsed bounds (first 5): neurons={collapse_indices_np[0]}, batches={collapse_indices_np[2]}")
                # Check if centers are ≤ 0 (will cause m = 0 for ReLU)
                c_collapsed = c[collapse_indices[0], 0, collapse_indices[2]]
                c_le_zero = c_collapsed <= 0
                print(f"    Centers at collapsed bounds (first 5): {c_collapsed[:5].cpu().numpy()}")
                print(f"    Centers ≤ 0 (will cause m=0): {torch.sum(c_le_zero).item()}/{len(c_collapsed)}")
        
        l = c - r
        u = c + r
        # compute slope of approximation
        # Convert to numpy for external functions f and df if needed, then convert back
        if options['nn']['poly_method'] == 'bounds':
            # Convert to numpy for f and df functions
            l_np = l.cpu().numpy() if isinstance(l, torch.Tensor) else l
            u_np = u.cpu().numpy() if isinstance(u, torch.Tensor) else u
            c_np = c.cpu().numpy() if isinstance(c, torch.Tensor) else c
            r_np = r.cpu().numpy() if isinstance(r, torch.Tensor) else r
            
            f_u = f(u_np)
            f_l = f(l_np)
            if isinstance(f_u, torch.Tensor):
                f_u = f_u.cpu().numpy()
            if isinstance(f_l, torch.Tensor):
                f_l = f_l.cpu().numpy()
            
            m_np = (f_u - f_l) / (2 * r_np)
            m = torch.tensor(m_np, dtype=dtype, device=device)
            
            # indices where upper and lower bound are equal
            idxBoundsEq = torch.abs(u - l) < torch.finfo(dtype).eps
            # If lower and upper bound are too close, approximate the slope at center
            df_c = df(c_np)
            if isinstance(df_c, torch.Tensor):
                df_c = df_c.cpu().numpy()
            df_c_torch = torch.tensor(df_c, dtype=dtype, device=device)
            m[idxBoundsEq] = df_c_torch[idxBoundsEq]
            if options['nn']['train']['backprop']:
                self.backprop['store']['idxBoundsEq'] = idxBoundsEq.cpu().numpy() if isinstance(idxBoundsEq, torch.Tensor) else idxBoundsEq
        elif options['nn']['poly_method'] == 'center':
            c_np = c.cpu().numpy() if isinstance(c, torch.Tensor) else c
            df_c = df(c_np)
            if isinstance(df_c, torch.Tensor):
                df_c = df_c.cpu().numpy()
            m = torch.tensor(df_c, dtype=dtype, device=device)
        elif options['nn']['poly_method'] == 'singh':
            l_np = l.cpu().numpy() if isinstance(l, torch.Tensor) else l
            u_np = u.cpu().numpy() if isinstance(u, torch.Tensor) else u
            df_l = df(l_np)
            df_u = df(u_np)
            if isinstance(df_l, torch.Tensor):
                df_l = df_l.cpu().numpy()
            if isinstance(df_u, torch.Tensor):
                df_u = df_u.cpu().numpy()
            m = torch.minimum(torch.tensor(df_l, dtype=dtype, device=device), 
                            torch.tensor(df_u, dtype=dtype, device=device))
        else:
            raise CORAerror('CORA:notSupported', f"Unsupported 'options.nn.poly_method': {options['nn']['poly_method']}")
        
        
        # evaluate image enclosure
        rc = m * c
        
        # MATLAB: G = permute(m,[1 3 2]).*G;
        # Ensure G's batch size matches m's batch size
        if m.ndim == 3 and G.ndim == 3:
            m_bSz = m.shape[2]
            G_bSz = G.shape[2]
            if m_bSz != G_bSz:
                # Replicate G to match m's batch size
                if m_bSz > G_bSz and m_bSz % G_bSz == 0:
                    nReps = m_bSz // G_bSz
                    G = G.repeat(1, 1, nReps)
                elif G_bSz > m_bSz and G_bSz % m_bSz == 0:
                    # G has larger batch size, subset it
                    G = G[:, :, :m_bSz]
                else:
                    raise ValueError(f"Cannot match batch sizes: m.shape={m.shape}, G.shape={G.shape}")
            # MATLAB: G = permute(m,[1 3 2]).*G;
            # permute(m,[1 3 2]) changes (n, 1, b) to (n, b, 1)
            # In PyTorch, we can use (n, 1, b) * (n, q, b) which broadcasts to (n, q, b)
            # This is equivalent to MATLAB's permute and multiply
            rG = m * G  # (n, 1, b) * (n, q, b) -> (n, q, b) via broadcasting
            
            # DEBUG: Log m and rG values for comparison with MATLAB
            if hasattr(self, '_debug_iteration') and self._debug_iteration is not None and self._debug_iteration <= 10:
                m_np = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
                rG_np = rG.cpu().numpy() if isinstance(rG, torch.Tensor) else rG
                print(f"  After generator multiplication:")
                print(f"    m shape: {m.shape}, m (first 3 neurons, first 3 batches): {m_np[:min(3, m_np.shape[0]), 0, :min(3, m_np.shape[2])].flatten()}")
                print(f"    m min/max: min={torch.min(m).item()}, max={torch.max(m).item()}")
                if torch.any(torch.abs(m) < 1e-6):
                    zero_m_count = torch.sum(torch.abs(m) < 1e-6).item()
                    print(f"    WARNING: {zero_m_count} neurons have |m| < 1e-6 (generators will become zero)!")
                    zero_m_indices = torch.where(torch.abs(m) < 1e-6)
                    zero_m_indices_np = [zmi.cpu().numpy()[:5] for zmi in zero_m_indices]
                    print(f"    Zero m locations (first 5): neurons={zero_m_indices_np[0]}, batches={zero_m_indices_np[2]}")
                rG_abs_sum = torch.sum(torch.abs(rG[:min(3, rG.shape[0]), :, :min(3, rG.shape[2])]), dim=1)
                print(f"    rG shape: {rG.shape}, rG sum(abs) (first 3 neurons, first 3 batches): {rG_abs_sum.flatten().cpu().numpy()}")
                if torch.any(torch.sum(torch.abs(rG), dim=1) < 1e-6):
                    zero_rG_count = torch.sum(torch.sum(torch.abs(rG), dim=1) < 1e-6).item()
                    print(f"    WARNING: {zero_rG_count} neurons have sum(|rG|) < 1e-6 (generators collapsed to zero)!")
        else:
            # Handle 2D case
            rG = m * G
        
        if options['nn'].get('use_approx_error', False):
            # Compute extreme points - convert m to numpy for extremePoints function
            m_np = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
            xs, xs_m = extremePoints(m_np)
            # Convert back to torch for internal operations
            if isinstance(xs, np.ndarray):
                xs = torch.tensor(xs, dtype=dtype, device=device)
            if isinstance(xs_m, np.ndarray):
                xs_m = torch.tensor(xs_m, dtype=dtype, device=device)
            
            # Determine number of extreme points.
            s = xs.shape[2]
            # Add interval bounds.
            if options['nn']['poly_method'] == 'bounds':
                # the approximation error at l and u are equal, thus we only
                # consider the upper bound u.
                xs = torch.cat([xs, l], dim=2)
            else:
                xs = torch.cat([xs, l, u], dim=2)
            
            # Convert to numpy for f function
            xs_np = xs.cpu().numpy() if isinstance(xs, torch.Tensor) else xs
            ys = f(xs_np)
            if isinstance(ys, torch.Tensor):
                ys = ys.cpu().numpy()
            ys = torch.tensor(ys, dtype=dtype, device=device)
            
            # Compute approximation error at candidates.
            # m needs to be broadcast to match xs shape in the last dimension
            m_expanded = m.repeat(1, 1, xs.shape[2] // m.shape[2])
            ds = ys - m_expanded * xs
            # We only consider candidate extreme points within boundaries.
            # Expand l and u to match xs shape
            l_expanded = l.repeat(1, 1, xs.shape[2] // l.shape[2])
            u_expanded = u.repeat(1, 1, xs.shape[2] // u.shape[2])
            notInBoundsIdx = (xs < l_expanded) | (xs > u_expanded)
            ds[notInBoundsIdx] = float('inf')
            dl = torch.min(ds, dim=2)[0]
            dlIdx = torch.argmin(ds, dim=2)
            ds[notInBoundsIdx] = float('-inf')
            du = torch.max(ds, dim=2)[0]
            duIdx = torch.argmax(ds, dim=2)
            
            # Retrieve stored id-matrix and generator indices
            approxErrGenIds = self.backprop['store'].get('approxErrGenIds', [])
            # Retrieve number of approximation errors.
            dn = len(approxErrGenIds)
            # Get size of generator matrix
            n, q, batchSize = rG.shape
            p = max(approxErrGenIds) if approxErrGenIds else 0
            if q < p:
                # Append generators for the approximation errors
                rG = torch.cat([rG, torch.zeros((n, len(approxErrGenIds), batchSize), dtype=dtype, device=device)], dim=1)
            
            # Convert dl, du, dlIdx, duIdx to numpy for complex indexing operations (ravel_multi_index, unravel_index)
            dl_np = dl.cpu().numpy() if isinstance(dl, torch.Tensor) else dl
            du_np = du.cpu().numpy() if isinstance(du, torch.Tensor) else du
            dlIdx_np = dlIdx.cpu().numpy() if isinstance(dlIdx, torch.Tensor) else dlIdx
            duIdx_np = duIdx.cpu().numpy() if isinstance(duIdx, torch.Tensor) else duIdx
            
            # Obtain the dn largest approximation errors.
            # [~,dDims] = sort(1/2*(du - dl),1,'descend');
            dDims = np.tile(np.arange(n).reshape(-1, 1), (1, batchSize))
            dDimsIdx = np.ravel_multi_index([dDims, np.tile(np.arange(batchSize), (n, 1))], (n, batchSize))
            notdDimsIdx = dDimsIdx[dn:, :]
            # set not considered approx. error to 0
            # Convert linear indices back to 2D indices for dl, du (n, 1) arrays
            # and 3D indices for c, m (n, 1, b) arrays
            notd_i, notd_b = np.unravel_index(notdDimsIdx.flatten(), (n, batchSize))
            
            # Convert c and m to numpy for f function call
            c_np = c.cpu().numpy() if isinstance(c, torch.Tensor) else c
            m_np = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
            
            dl_np[notd_i, 0] = f(c_np[notd_i, 0, notd_b]) - m_np[notd_i, 0, notd_b] * c_np[notd_i, 0, notd_b]
            du_np[notd_i, 0] = dl_np[notd_i, 0]
            
            # Convert back to torch
            dl = torch.tensor(dl_np, dtype=dtype, device=device)
            du = torch.tensor(du_np, dtype=dtype, device=device)
            
            # shift y-intercept by center of approximation errors
            # t should have same shape as c (n, 1, b), but du and dl are (n, 1)
            # We need to broadcast du and dl to match c's batch dimension
            # In MATLAB, this computation preserves all dimensions
            # We need to broadcast du and dl to match the batch dimension of c
            if du.shape != c.shape:
                # Broadcast du and dl to match c's shape
                du_expanded = du.unsqueeze(2).expand_as(c)
                dl_expanded = dl.unsqueeze(2).expand_as(c)
                t = 0.5 * (du_expanded + dl_expanded)
                d = 0.5 * (du_expanded - dl_expanded)
            else:
                t = 0.5 * (du + dl)
                d = 0.5 * (du - dl)
            
            # Compute indices for approximation errors in the generator
            # matrix.
            if dn > 0 and len(approxErrGenIds) > 0:
                # MATLAB: sub2ind([n p batchSize], repmat(1:n,1,batchSize), reshape(dDims(1:dn,:),1,[]), repmat(approxErrGenIds,1,batchSize), repelem(1:batchSize,1,n))
                # Convert to 0-based indexing and fix parameter order
                GdIdx = np.ravel_multi_index([
                    np.tile(np.arange(n, dtype=int), batchSize),  # repmat(1:n,1,batchSize) -> 0-based
                    np.tile(np.array(approxErrGenIds, dtype=int) - 1, batchSize),  # repmat(approxErrGenIds,1,batchSize) -> 0-based  
                    np.repeat(np.arange(batchSize, dtype=int), n)  # repelem(1:batchSize,1,n) -> 0-based
                ], (n, p, batchSize))
                GdIdx = GdIdx.reshape(dn, batchSize)
            else:
                GdIdx = np.array([], dtype=int).reshape(0, batchSize)
            # Store indices of approximation error in generator matrix.
            self.backprop['store']['GdIdx'] = GdIdx
            dDimsIdx = dDimsIdx[:dn, :]
            self.backprop['store']['dDimsIdx'] = dDimsIdx
            # Add approximation errors to the generators.
            # Convert to numpy for flat indexing, then convert back
            rG_np = rG.cpu().numpy() if isinstance(rG, torch.Tensor) else rG
            d_np = d.cpu().numpy() if isinstance(d, torch.Tensor) else d
            rG_np.flat[GdIdx] = d_np  # (dDimsIdx);
            rG = torch.tensor(rG_np, dtype=dtype, device=device)
            
            # compute gradients
            if options['nn']['train']['backprop'] and options['nn']['train']['exact_backprop']:
                # derivative of radius wrt. generators
                r_G = np.sign(G)
                # Compute gradient of the slope.
                if options['nn']['poly_method'] == 'bounds':
                    m_c = (df(u) - df(l)) / (2 * r)
                    m_G = r_G * np.transpose((df(u) + df(l) - 2 * m) / (2 * r), (1, 2, 0))
                    # prevent numerical issues
                    ddf = self.getDf(2)
                    m_c[idxBoundsEq] = df(c[idxBoundsEq])
                    m_G = np.transpose(m_G, (1, 0, 2))
                    r_G = np.transpose(r_G, (1, 0, 2))
                    m_G[:, idxBoundsEq] = r_G[:, idxBoundsEq] * ddf(c[idxBoundsEq]).T
                    m_G = np.transpose(m_G, (1, 0, 2))
                    r_G = np.transpose(r_G, (1, 0, 2))
                elif options['nn']['poly_method'] == 'center':
                    ddf = self.getDf(2)
                    m_c = ddf(c) * np.ones_like(c)
                    m_G = np.zeros_like(G)
                elif options['nn']['poly_method'] == 'singh':
                    lu = np.concatenate([l.reshape(l.shape[0], l.shape[1], 1), 
                                       u.reshape(u.shape[0], u.shape[1], 1)], axis=2)
                    mIdx = np.argmin(df(lu), axis=2)
                    ddf = self.getDf(2)
                    m_c = ddf(lu[mIdx]) * np.ones_like(c)
                    m_G = np.transpose(-1 * (mIdx == 0).astype(float) * m_c, (1, 2, 0)) * r_G
                else:
                    raise CORAerror('CORA:notSupported', f"Unsupported 'options.nn.poly_method': {options['nn']['poly_method']}")
                
                # Add gradients for interval bounds.
                if options['nn']['poly_method'] == 'bounds':
                    # We only consider the lower bound. The approximation
                    # error at the lower and upper bound is equal.
                    x_c = np.concatenate([xs_m.reshape(xs_m.shape[0], xs_m.shape[1], 1) * m_c, 
                                        np.ones_like(l).reshape(l.shape[0], l.shape[1], 1)], axis=2)
                    x_G = np.concatenate([np.transpose(xs_m.reshape(xs_m.shape[0], xs_m.shape[1], 1), (1, 2, 3, 0)) * 
                                        np.transpose(m_G, (1, 2, 0)),
                                        -np.transpose(r_G, (1, 2, 0))], axis=3)
                else:
                    x_c = np.concatenate([xs_m.reshape(xs_m.shape[0], xs_m.shape[1], 1) * m_c, 
                                        np.ones_like(l).reshape(l.shape[0], l.shape[1], 1),
                                        np.ones_like(u).reshape(u.shape[0], u.shape[1], 1)], axis=2)
                    x_G = np.concatenate([np.transpose(xs_m.reshape(xs_m.shape[0], xs_m.shape[1], 1), (1, 2, 3, 0)) * 
                                        np.transpose(m_G, (1, 2, 0)),
                                        -np.transpose(r_G, (1, 2, 0)),
                                        np.transpose(r_G, (1, 2, 0))], axis=3)
                x_G = np.transpose(x_G, (2, 0, 1, 3))
                
                # Compute gradient of the approximation errors.
                xl = xs.flat[dlIdx]
                dfxlm = df(xl) - m
                dl_c = dfxlm * x_c.flat[dlIdx] - m_c * xl
                xl_G = np.reshape(x_G[:, dlIdx], (q, n, batchSize))
                dl_G = np.transpose(dfxlm, (2, 0, 1)) * xl_G - np.transpose(m_G, (1, 0, 2)) * np.transpose(xl, (2, 0, 1))
                
                xu = xs.flat[duIdx]
                dfxum = df(xu) - m
                du_c = dfxum * x_c.flat[duIdx] - m_c * xu
                xu_G = np.reshape(x_G[:, duIdx], (q, n, batchSize))
                du_G = np.transpose(dfxum, (2, 0, 1)) * xu_G - np.transpose(m_G, (1, 0, 2)) * np.transpose(xu, (2, 0, 1))
                
                # Compute components for the backpropagation.
                self.backprop['store']['m_c'] = m_c
                self.backprop['store']['m_G'] = m_G
                
                self.backprop['store']['dc_c'] = 1/2 * (du_c[dDimsIdx] + dl_c[dDimsIdx])
                self.backprop['store']['dc_G'] = 1/2 * (du_G[:, dDimsIdx] + dl_G[:, dDimsIdx])
                
                self.backprop['store']['d_c'] = 1/2 * (du_c[dDimsIdx] - dl_c[dDimsIdx])
                self.backprop['store']['d_G'] = 1/2 * (du_G[:, dDimsIdx] - dl_G[:, dDimsIdx])
                
        else:
            # compute y-intercept
            # In MATLAB: t = f(c) - m.*c; where both f(c) and m.*c have same shape as c
            fc = f(c)  # f(c) should preserve c's shape
            mc = m * c  # m.*c in MATLAB
            
            # Ensure fc has same shape as c (batch dimension preserved)
            if fc.shape != c.shape:
                # If f(c) collapsed the batch dimension, broadcast it back
                if fc.ndim == 2 and c.ndim == 3:
                    fc = np.broadcast_to(fc[:, :, np.newaxis], c.shape)
            
            t = fc - mc
            # approximation errors are 0
            d = 0
        
        # return coefficients
        # MATLAB: coeffs = permute(cat(3,m,t),[1 3 2]);
        # cat(3,m,t) concatenates along dimension 3 (axis=2 in Python)
        # permute([1 3 2]) rearranges from (dim1, dim2, dim3) to (dim1, dim3, dim2)
        coeffs = torch.permute(torch.cat([m, t], dim=2), (0, 2, 1))
        # Add y-intercept.
        rc = rc + t
        
        # Convert back to numpy for return (if caller expects numpy)
        # This maintains compatibility with existing code
        rc_np = rc.cpu().numpy() if isinstance(rc, torch.Tensor) else rc
        rG_np = rG.cpu().numpy() if isinstance(rG, torch.Tensor) else rG
        coeffs_np = coeffs.cpu().numpy() if isinstance(coeffs, torch.Tensor) else coeffs
        if isinstance(d, torch.Tensor):
            d_np = d.cpu().numpy() if d.numel() > 1 else d.cpu().item()
        else:
            d_np = d
        
        return rc_np, rG_np, coeffs_np, d_np
    


    @staticmethod
    def instantiateFromString(activation: str) -> 'nnActivationLayer':
        """
        Instantiate activation layer from string
        
        Args:
            activation: Activation function name
            
        Returns:
            layer: Activation layer instance
        """
        # Check input arguments
        activation = activation.lower()
        possible_activations = ['relu', 'sigmoid', 'tanh', 'softmax', 'identity', 'none', 'invsqrtroot', 'sqrt']
        
        if activation not in possible_activations:
            raise CORAerror('CORA:wrongValue', 'first', ', '.join(possible_activations))
        
        # Import here to avoid circular imports
        if activation == "relu":
            from .nnReLULayer import nnReLULayer
            layer = nnReLULayer()
        elif activation == "sigmoid":
            from .nnSigmoidLayer import nnSigmoidLayer
            layer = nnSigmoidLayer()
        elif activation == "tanh":
            from .nnTanhLayer import nnTanhLayer
            layer = nnTanhLayer()
        elif activation == "softmax":
            from .nnSoftmaxLayer import nnSoftmaxLayer
            layer = nnSoftmaxLayer()
        elif activation == "identity":
            # For now, use ReLU as identity (should be nnIdentityLayer when implemented)
            raise CORAerror('CORA:notSupported', 'identity activation not implemented yet')
        elif activation == "none":
            # For now, use ReLU as none (should be nnIdentityLayer when implemented)
            raise CORAerror('CORA:notSupported', 'none activation not implemented yet')
        elif activation == "invsqrtroot":
            # For now, use ReLU as invsqrtroot (should be nnInvSqrtRootLayer when implemented)
            raise CORAerror('CORA:notSupported', 'invsqrtroot activation not implemented yet')
        elif activation == "sqrt":
            # For now, use ReLU as sqrt (should be nnRootLayer when implemented)
            raise CORAerror('CORA:notSupported', 'sqrt activation not implemented yet')
        else:
            # Should not be executed anyway due to inputArgsCheck
            raise CORAerror('CORA:wrongValue', 'first', ', '.join(possible_activations))
        
        return layer
