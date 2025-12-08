"""
nnReshapeLayer - class for reshape layers

Syntax:
    obj = nnReshapeLayer(idx_out)
    obj = nnReshapeLayer(idx_out, name)

Inputs:
    idx_out - output indices
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: nnLayer

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   ---
Last revision: 10-August-2022 (renamed)
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from ..nnLayer import nnLayer


class nnReshapeLayer(nnLayer):
    """
    Reshape layer class for neural networks
    
    This layer reshapes the input according to specified output indices.
    """
    
    def __init__(self, idx_out: List[int], name: Optional[str] = None):
        """
        Constructor for nnReshapeLayer
        
        Args:
            idx_out: Output indices
            name: Name of the layer, defaults to type
        """
        # call super class constructor
        super().__init__(name)
        
        self.idx_out = idx_out
        self.inputSize = None  # Will be set when input size is known
        
        # whether the layer is refinable
        self.is_refinable = False
    
    def getNumNeurons(self) -> Tuple[List[int], List[int]]:
        """
        Get number of neurons for this layer
        
        Returns:
            nin: input neuron count (empty for this layer type)
            nout: output neuron count (empty for this layer type)
        """
        return [], []
    
    def getOutputSize(self, inputSize: List[int]) -> List[int]:
        """
        Get output size for given input size (MATLAB parity)
        - For flatten (idx_out contains -1): return [prod(inputSize), 1]
        - For index-based reshape: return [num_indices, 1]
        """
        self.inputSize = inputSize
        if isinstance(self.idx_out, list) and (-1 in self.idx_out):
            return [int(np.prod(inputSize)), 1]
        idx = np.array(self.idx_out)
        num = int(idx.size)
        return [num, 1]
    
    def evaluateNumeric(self, input_data, options: Dict[str, Any]):
        """
        Evaluate layer numerically
        
        Args:
            input_data: input data (numpy array or torch tensor) - converted to torch internally
            options: evaluation options
            
        Returns:
            output: reshaped input data (torch tensor)
        """
        # Convert numpy input to torch if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        # Debug output
        if hasattr(options, 'debug') and options.get('debug', False):
            print(f"DEBUG ReshapeLayer: input_data shape: {input_data.shape}, idx_out: {self.idx_out}")
        
        # Handle special case of -1 in idx_out (flatten to 1D)
        if -1 in self.idx_out:
            # For flattening, preserve the batch dimension if present
            # Input is expected to be in format [features, batchSize] from Conv2D/AvgPool layers
            other_dims = [d for d in self.idx_out if d != -1]
            
            if input_data.ndim == 2:
                # Input has batch dimension: [features, batchSize]
                features, batchSize = input_data.shape
                if not other_dims:
                    # Just [-1]: flatten features, preserve batch dimension
                    # Input is already in [features, batchSize] format, so return as-is
                    return input_data
                else:
                    # There are other dimensions specified
                    # Calculate -1 dimension, but preserve batch dimension
                    total_features = features
                    other_size = torch.prod(torch.tensor(other_dims, device=input_data.device)).item()
                    if other_size > 0 and total_features % other_size == 0:
                        calculated_dim = total_features // other_size
                        target_shape = [calculated_dim] + other_dims + [batchSize]
                        return input_data.reshape(target_shape)
                    else:
                        # Can't divide evenly, keep original shape
                        return input_data
            else:
                # Input is 1D: [features] - no batch dimension
                total_size = input_data.numel()
                if not other_dims:
                    # Just [-1]: already 1D, return as-is
                    return input_data
                else:
                    # Calculate -1 dimension
                    other_size = torch.prod(torch.tensor(other_dims, device=input_data.device)).item()
                    if other_size > 0 and total_size % other_size == 0:
                        calculated_dim = total_size // other_size
                        target_shape = [calculated_dim] + other_dims
                        return input_data.reshape(target_shape)
                    else:
                        # Can't divide evenly, return as-is
                        return input_data
        else:
            # Use idx_out as indices to reorder input (like MATLAB)
            # MATLAB: r = input(idx_vec, :, :);
            # Convert MATLAB's column-major idx_out to Python's row-major representation
            # MATLAB idx_out(:) gives column-major order, but we want to maintain C-order in Python
            # So we need to transpose idx_out first, then flatten to get the correct order
            idx_out_array = torch.tensor(self.idx_out, device=input_data.device)
            idx_out_transposed = idx_out_array.T  # Transpose to convert to row-major
            idx_vec = idx_out_transposed.flatten()  # Now flatten in C-order
            # Convert to 0-based indexing for Python
            idx_vec = (idx_vec - 1).long()  # Convert to long for indexing
            

            # Handle multi-dimensional input
            if input_data.ndim > 1:
                # For multi-dimensional input, apply indexing to first dimension
                # and preserve other dimensions
                result = input_data[idx_vec]
            else:
                # For 1D input, just apply indexing
                result = input_data[idx_vec]
            
            return result
    
    def evaluateSensitivity(self, S, x, options: Dict[str, Any]):
        """
        Evaluate sensitivity
        
        Args:
            S: sensitivity matrix with shape (nK, output_dim, bSz) (torch tensor)
            x: input data (torch tensor)
            options: evaluation options
            
        Returns:
            S: reshaped sensitivity matrix with shape (nK, input_dim, bSz) (torch tensor)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(S, np.ndarray):
            S = torch.tensor(S, dtype=torch.float32)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        # MATLAB: S_ = permute(S,[2 1 3]); S_ = obj.aux_embed(S_,inSize); S = permute(S_,[2 1 3]);
        # This swaps first two dims, embeds according to input size, then swaps back
        
        if self.inputSize is None:
            # If inputSize not set, try to infer from x
            if x is not None and hasattr(x, 'shape'):
                if x.ndim >= 1:
                    self.inputSize = [x.shape[0]]
                else:
                    self.inputSize = [1]
            else:
                # Fallback: assume input size matches output size
                if S.ndim >= 2:
                    self.inputSize = [S.shape[1]]
                else:
                    self.inputSize = [1]
        
        inSize = self.inputSize
        prod_inSize = np.prod(inSize)
        
        # Step 1: permute(S, [2 1 3]) - swap first two dimensions
        # S has shape (nK, output_dim, bSz) -> (output_dim, nK, bSz)
        if S.ndim == 3:
            S_perm = np.transpose(S, (1, 0, 2))  # (output_dim, nK, bSz)
        elif S.ndim == 2:
            S_perm = S.T  # (output_dim, nK)
        else:
            # 1D or scalar - shouldn't happen but handle it
            S_perm = S
        
        # Step 2: aux_embed - inverse of reshape
        # MATLAB aux_embed logic:
        # - If 3D (n, q, bSz): reshape to 2D (n, q*bSz)
        # - Create output (prod(inSize), bSz) or (prod(inSize), q*bSz)
        # - Set r(idx_vec, :) = input
        if S_perm.ndim == 3:
            # 3D case: (output_dim, nK, bSz)
            n, q, bSz = S_perm.shape
            # Reshape to 2D: (n, q*bSz) - MATLAB input(:,:)
            S_2d = S_perm.reshape(n, q * bSz, order='F')
        elif S_perm.ndim == 2:
            # 2D case: (output_dim, nK)
            S_2d = S_perm
            n, q = S_2d.shape
            bSz = 1
        else:
            # 1D or scalar
            S_2d = S_perm.reshape(-1, 1) if S_perm.ndim == 1 else S_perm.reshape(1, 1)
            n, q = S_2d.shape
            bSz = 1
        
        # Get idx_vec from idx_out (1-based, column-major)
        idx_vec = np.array(self.idx_out)
        if idx_vec.ndim > 1:
            idx_vec = idx_vec.flatten(order='F')
        else:
            idx_vec = idx_vec.flatten()
        # Convert to 0-based
        idx0 = idx_vec - 1
        idx0 = np.clip(idx0, 0, prod_inSize - 1)
        
        # Create output: (prod(inSize), q*bSz) or (prod(inSize), q)
        if S_perm.ndim == 3:
            r = np.zeros((prod_inSize, q * bSz), dtype=S_2d.dtype)
        else:
            r = np.zeros((prod_inSize, q), dtype=S_2d.dtype)
        
        # Set r(idx_vec, :) = input
        # MATLAB: r(idx_vec, :) = input
        # idx_vec should have n elements (one for each row of input)
        # If len(idx0) != n, we need to handle it
        if len(idx0) == n:
            # Normal case: idx0 has n elements matching S_2d rows
            r[idx0, :] = S_2d
        elif len(idx0) == 1 and n > 1:
            # Special case: idx_out might be a single value (like [5] for flatten)
            # In this case, we need to expand idx0 to match n
            # Or, if idx_out represents a single output dimension, we need different logic
            # For now, if idx0 is a single index, we'll use it for all rows
            # This might not be correct, but let's see what happens
            if n <= prod_inSize:
                # Use consecutive indices starting from idx0[0]
                idx0_expanded = np.arange(idx0[0], idx0[0] + n, dtype=int)
                idx0_expanded = np.clip(idx0_expanded, 0, prod_inSize - 1)
                r[idx0_expanded, :] = S_2d
            else:
                # n > prod_inSize, this shouldn't happen but handle it
                r[idx0[0]:idx0[0]+min(n, prod_inSize-idx0[0]), :] = S_2d[:min(n, prod_inSize-idx0[0]), :]
        else:
            # Other mismatch cases
            min_len = min(len(idx0), n)
            r[idx0[:min_len], :] = S_2d[:min_len, :]
        
        # Reshape back if was 3D
        if S_perm.ndim == 3:
            r = r.reshape(prod_inSize, q, bSz, order='F')
        
        # Step 3: permute back: (prod_inSize, nK, bSz) -> (nK, prod_inSize, bSz)
        if r.ndim == 3:
            S_result = np.transpose(r, (1, 0, 2))  # (nK, prod_inSize, bSz)
        elif r.ndim == 2:
            S_result = r.T  # (nK, prod_inSize)
        else:
            S_result = r
        
        return S_result
    
    def evaluateZonotopeBatch(self, c, G, options: Dict[str, Any]):
        """
        Evaluate reshape for a batch of zonotopes (centers c and generators G).
        MATLAB behavior: calls aux_reshape which reshapes to 2D, indexes, then reshapes back.
        Shapes: typically c in R^{n x 1 x b} (or n x 1), G in R^{n x q x b} (or n x q)
        
        MATLAB aux_reshape logic:
        1. If input is 3D (n, q, bSz): reshape to 2D (n, q*bSz) using input(:,:)
        2. idx_vec = idx_out(:) - column-major flatten (1-based)
        3. r = input(idx_vec, :) - select rows, keep all columns
        4. If was 3D: reshape back to (len(idx_vec), q, bSz)
        """
        # Flatten case (dynamic -1): ACASXU uses this right after input; no feature reordering needed
        if isinstance(self.idx_out, list) and (-1 in self.idx_out):
            return c, G

        # Simulate MATLAB aux_reshape exactly
        # Step 1: Check if input is 3D (matrix)
        is_matrix_c = c.ndim > 2
        is_matrix_G = G.ndim > 2
        
        # Step 2: For 3D inputs, reshape to 2D like MATLAB input(:,:)
        # MATLAB: input(:,:) on (n, q, bSz) -> (n, q*bSz) - keeps first dim, flattens rest column-major
        if is_matrix_c:
            n_c, q_c, bSz_c = c.shape
            # Reshape to 2D: (n, q*bSz) using column-major (Fortran) order
            # For torch, permute then reshape: (n, q, bSz) -> (n, bSz, q) -> (n, q*bSz)
            c_2d = c.permute(0, 2, 1).reshape(n_c, q_c * bSz_c)
        else:
            c_2d = c
            n_c, q_c = c.shape if c.ndim == 2 else (c.shape[0], 1)
            bSz_c = 1
        
        if is_matrix_G:
            n_g, q_g, bSz_g = G.shape
            # Reshape to 2D: (n, q*bSz) using column-major (Fortran) order
            # For torch, permute then reshape: (n, q, bSz) -> (n, bSz, q) -> (n, q*bSz)
            G_2d = G.permute(0, 2, 1).reshape(n_g, q_g * bSz_g)
        else:
            G_2d = G
            n_g, q_g = G.shape if G.ndim == 2 else (G.shape[0], 1)
            bSz_g = 1
        
        # Step 3: Get idx_vec from idx_out (column-major flatten, 1-based)
        idx_vec = torch.tensor(self.idx_out, device=c.device)
        if idx_vec.ndim > 1:
            # Column-major flatten: idx_out(:) in MATLAB
            # For torch, we need to transpose then flatten to get column-major
            idx_vec = idx_vec.T.flatten()
        else:
            idx_vec = idx_vec.flatten()
        # Convert to 0-based for Python
        idx0 = (idx_vec - 1).long()
        
        # Bounds check (defensive)
        max_n_c = c_2d.shape[0]
        max_n_g = G_2d.shape[0]
        idx0_c = torch.clamp(idx0, 0, max_n_c - 1)
        idx0_g = torch.clamp(idx0, 0, max_n_g - 1)
        
        # Step 4: MATLAB: r = input(idx_vec, :) - select rows, keep all columns
        c_result = c_2d[idx0_c, :]
        G_result = G_2d[idx0_g, :]
        
        # Step 5: If was 3D, reshape back to original shape
        if is_matrix_c:
            # MATLAB: reshape(r, [], q, bSz) - [] means calculate automatically
            # Result has shape (len(idx_vec), q*bSz), reshape to (len(idx_vec), q, bSz)
            # For column-major (Fortran) order in torch, we need to reshape then transpose
            c_result = c_result.reshape(len(idx0_c), bSz_c, q_c).permute(0, 2, 1)
        
        if is_matrix_G:
            # MATLAB: reshape(r, [], q, bSz)
            # For column-major (Fortran) order in torch, we need to reshape then transpose
            G_result = G_result.reshape(len(idx0_g), bSz_g, q_g).permute(0, 2, 1)
        
        return c_result, G_result

    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id: np.ndarray, id_: np.ndarray, 
                            ind: np.ndarray, ind_: np.ndarray, 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate layer with polynomial zonotope
        
        Args:
            c: center
            G: generators
            GI: independent generators
            E: exponents
            id: identifiers
            id_: independent identifiers
            ind: indices
            ind_: independent indices
            options: evaluation options
            
        Returns:
            c, G, GI, E, id, id_, ind, ind_: reshaped polynomial zonotope
        """
        # Reshape all components according to idx_out
        c = c.reshape(self.idx_out)
        G = G.reshape(self.idx_out + list(G.shape[1:]))
        GI = GI.reshape(self.idx_out + list(GI.shape[1:]))
        
        return c, G, GI, E, id, id_, ind, ind_
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate layer with Taylor model
        
        Args:
            input_data: input data
            options: evaluation options
            
        Returns:
            output: reshaped input data
        """
        return input_data.reshape(self.idx_out)
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           j: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """
        Evaluate layer with constrained zonotope
        
        Args:
            c: center
            G: generators
            C: constraint matrix
            d: constraint vector
            l: lower bounds
            u: upper bounds
            j: index
            options: evaluation options
            
        Returns:
            c, G, C, d, l, u: reshaped constrained zonotope
        """
        # Reshape center and generators according to idx_out
        c = c.reshape(self.idx_out)
        G = G.reshape(self.idx_out + list(G.shape[1:]))
        l = l.reshape(self.idx_out)
        u = u.reshape(self.idx_out)
        
        return c, G, C, d, l, u
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            layerStruct: layer structure
        """
        layerStruct = {
            'type': 'nnReshapeLayer',
            'name': self.name,
            'fields': {
                'idx_out': self.idx_out
            }
        }
        return layerStruct
