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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import abstractmethod
from ..nnLayer import nnLayer

from cora_python.contSet.interval import Interval
from cora_python.contSet.emptySet import EmptySet


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
        
        # function handles
        self.f = None  # function
        self.df = None  # function derivative
        
        # adaptive refinement
        self.order = 1  # order of approximation polynomial
        self.refine_heu = None  # heuristic for refinement
        self.do_refinement = True  # whether the layer should be refined
        
        self.l = []  # lower bound of last input
        self.u = []  # upper bound of last input
        
        self.merged_neurons = []  # network reduction
        
        # init function handles
        self.f = lambda x: self.evaluateNumeric(x, {'backprop': False})
        self.df = self.getDf(1)
    
    # evaluate (element-wise) -------------------------------------------------
    
    def evaluateSensitivity(self, S: np.ndarray, x: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate sensitivity
        
        Args:
            S: Sensitivity matrix
            x: Input point
            options: Evaluation options
            
        Returns:
            S: Updated sensitivity matrix
        """
        S = S * self.df(x).transpose(2, 0, 1)
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
                not all(size == size_old for size, size_old in zip(bounds.shape, self.l.shape))):
                self.l = bounds.inf
                self.u = bounds.sup
                
                # set bounds
            elif (self._representsa_emptySet(bounds, eps=1e-10) and 
                  any(np.isnan(self.l)) and any(np.isnan(self.u))):
                bounds = Interval(self.l, self.u)
            
            self.l = np.maximum(self.l, bounds.inf)
            self.u = np.minimum(self.u, bounds.sup)
        
        # propagate through layer
        bounds = super().evaluateInterval(bounds, options)
        return bounds
    
    def evaluateZonotopeBatch(self, c: np.ndarray, G: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate zonotope batch
        
        Args:
            c: Center
            G: Generators
            options: Evaluation options
            
        Returns:
            Tuple of (rc, rG) results
        """
        # Compute image enclosure
        rc, rG, coeffs, d = self.aux_imgEncBatch(self.f, self.df, c, G, options, 
                                                 lambda m: self.computeExtremePointsBatch(m, options))
        
        # store inputs and coeffs for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            # Store coefficients
            self.backprop['store']['coeffs'] = coeffs
            
            # Store the slope.
            if options.get('nn', {}).get('train', {}).get('exact_backprop', False):
                # Store gradient for the backprop through an image
                # enclosure.
                if hasattr(self, 'm_l') and hasattr(self, 'm_u'):
                    self.backprop['store']['m_l'] = self.m_l
                    self.backprop['store']['m_u'] = self.m_u
                
                if options.get('nn', {}).get('use_approx_error', False):
                    if not options.get('nn', {}).get('interval_center', False):
                        if hasattr(self, 'GdIdx'):
                            self.backprop['store']['GdIdx'] = self.GdIdx
                    if hasattr(self, 'dDimsIdx'):
                        self.backprop['store']['dDimsIdx'] = self.dDimsIdx
                    if hasattr(self, 'notdDimsIdx'):
                        self.backprop['store']['notdDimsIdx'] = self.notdDimsIdx
                    
                    if hasattr(self, 'dl_l') and hasattr(self, 'dl_u'):
                        self.backprop['store']['dl_l'] = self.dl_l
                        self.backprop['store']['dl_u'] = self.dl_u
                    if hasattr(self, 'du_l') and hasattr(self, 'du_u'):
                        self.backprop['store']['du_l'] = self.du_l
                        self.backprop['store']['du_u'] = self.du_u
        
        return rc, rG
    
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
        raise NotImplementedError("conZonotope not supported for this layer")
    
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
        # obtain stored coefficients of image enclosure from forward prop.
        coeffs = self.backprop['store']['coeffs']
        # obtain slope of the approximation
        m = coeffs[:, 0, :] if len(coeffs.shape) == 3 else coeffs[:, 0]
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
            
            # Precompute outer product of gradients and inputs.
            hadProdc = np.transpose(gc * c, (1, 2, 0))
            hadProdG = gG[:, genIds, :] * G
            
            # Backprop gradients.
            rgc = gc * m + m_c * np.reshape(hadProdc + np.sum(hadProdG, axis=1), c.shape)
            rgc[dDimsIdx] = rgc[dDimsIdx] + dc_c * gc[dDimsIdx]
            rgc[dDimsIdx] = rgc[dDimsIdx] + d_c * gG[GdIdx]
            # Assign results.
            gc = rgc
            
            rgG = gG[:, genIds, :] * np.transpose(m, (1, 2, 0)) + m_G * (hadProdc + hadProdG)
            rgG = np.transpose(rgG, (1, 0, 2))
            rgG[:, dDimsIdx] = rgG[:, dDimsIdx] + dc_G * np.reshape(gc[dDimsIdx], (1, -1))
            rgG[:, dDimsIdx] = rgG[:, dDimsIdx] + d_G * np.reshape(gG[GdIdx], (1, -1))
            rgG = np.transpose(rgG, (1, 0, 2))
            # Assign results.
            gG = rgG
            
        else:
            # Consider the approximation as fixed. Use the slope of the
            # approximation for backpropagation
            gc = gc * m
            gG = gG[:, genIds, :] * np.transpose(m, (1, 2, 0))
        
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
        # This would require nnHelper.lookupDf
        # For now, return a placeholder
        raise NotImplementedError("getDf not implemented in base class")
    
    def getNumNeurons(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get number of input and output neurons
        
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
    
    # approximation polynomial + error
    
    def computeApproxError(self, l: np.ndarray, u: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute approximation error
        
        Args:
            l: Lower bounds
            u: Upper bounds
            coeffs: Polynomial coefficients
            
        Returns:
            Tuple of (coeffs, d) updated coefficients and error bound
        """
        # bound approximation error according to [1, Sec. 3.2]
        
        # compute the difference between activation function and quad. fit
        df_l, df_u = self.getDerBounds(l, u)
        # This would require nnHelper.minMaxDiffOrder
        # For now, use a simplified approach
        diffl = np.zeros_like(l)
        diffu = np.zeros_like(u)
        
        # change polynomial s.t. lower and upper error are equal
        diffc = (diffl + diffu) / 2
        coeffs[-1] = coeffs[-1] + diffc
        d = diffu - diffc  # error is radius then.
        
        return coeffs, d
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            fieldStruct: Field structure
        """
        fieldStruct = {}
        if self.merged_neurons:
            fieldStruct['merged_neurons'] = self.merged_neurons
        return fieldStruct
    
    @staticmethod
    def instantiateFromString(activation: str) -> 'nnActivationLayer':
        """
        Instantiate activation layer from string
        
        Args:
            activation: Activation function name
            
        Returns:
            layer: Activation layer instance
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("instantiateFromString not implemented in base class")
    
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
    
    def computeApproxPoly(self, l: np.ndarray, u: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        """
        Compute approximating polynomial
        
        Args:
            l: Lower bounds
            u: Upper bounds
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (coeffs, d) polynomial coefficients and error bound
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("computeApproxPoly not implemented in base class")
    
    def findRegionPolys(self, tol: float, order: int, l_max: np.ndarray, u_max: np.ndarray, 
                        pStart: np.ndarray, dStart: float, pEnd: np.ndarray, dEnd: float) -> np.ndarray:
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
            coeffs: Polynomial coefficients
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("findRegionPolys not implemented in base class")
    
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
        raise NotImplementedError("computeExtremePointsBatch not implemented in base class")
    
    def aux_imgEncBatch(self, f: callable, df: callable, c: np.ndarray, G: np.ndarray, 
                        options: Dict[str, Any], extremePoints: callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Auxiliary function for image enclosure batch
        
        Args:
            f: Function
            df: Derivative function
            c: Center
            G: Generators
            options: Options
            extremePoints: Extreme points function
            
        Returns:
            Tuple of (rc, rG, coeffs, d) results
        """
        # obtain indices of active generators
        genIds = self.backprop['store'].get('genIds', slice(None))
        # compute bounds
        r = np.reshape(np.sum(np.abs(G[:, genIds, :]), axis=1), c.shape)
        # r = max(eps('like',c),r); % prevent numerical instabilities
        l = c - r
        u = c + r
        # compute slope of approximation
        if options['nn']['poly_method'] == 'bounds':
            m = (f(u) - f(l)) / (2 * r)
            # indices where upper and lower bound are equal
            idxBoundsEq = np.abs(u - l) < np.finfo(c.dtype).eps
            # If lower and upper bound are too close, approximate the slope
            # at center.
            m[idxBoundsEq] = df(c[idxBoundsEq])
            if options['nn']['train']['backprop']:
                self.backprop['store']['idxBoundsEq'] = idxBoundsEq
        elif options['nn']['poly_method'] == 'center':
            m = df(c).astype(c.dtype)
        elif options['nn']['poly_method'] == 'singh':
            m = np.minimum(df(l).astype(l.dtype), df(u).astype(u.dtype))
        else:
            raise ValueError(f"Unsupported 'options.nn.poly_method': {options['nn']['poly_method']}")
        
        # evaluate image enclosure
        rc = m * c
        # rG(:,genIds,:) = permute(m,[1 3 2]).*G(:,genIds,:);
        rG = np.transpose(m, (1, 2, 0)) * G
        
        if options['nn'].get('use_approx_error', False):
            # Compute extreme points.
            xs, xs_m = extremePoints(m)
            # Determine number of extreme points.
            s = xs.shape[2]
            # Add interval bounds.
            if options['nn']['poly_method'] == 'bounds':
                # the approximation error at l and u are equal, thus we only
                # consider the upper bound u.
                xs = np.concatenate([xs, l.reshape(l.shape[0], l.shape[1], 1)], axis=2)
            else:
                xs = np.concatenate([xs, l.reshape(l.shape[0], l.shape[1], 1), 
                                   u.reshape(u.shape[0], u.shape[1], 1)], axis=2)
            ys = f(xs)
            # Compute approximation error at candidates.
            ds = ys - m.reshape(m.shape[0], m.shape[1], 1) * xs
            # We only consider candidate extreme points within boundaries.
            notInBoundsIdx = (xs < l.reshape(l.shape[0], l.shape[1], 1) | 
                             xs > u.reshape(u.shape[0], u.shape[1], 1))
            ds[notInBoundsIdx] = np.inf
            dl = np.min(ds, axis=2)
            dlIdx = np.argmin(ds, axis=2)
            ds[notInBoundsIdx] = -np.inf
            du = np.max(ds, axis=2)
            duIdx = np.argmax(ds, axis=2)
            
            # Retrieve stored id-matrix and generator indices
            approxErrGenIds = self.backprop['store'].get('approxErrGenIds', [])
            # Retrieve number of approximation errors.
            dn = len(approxErrGenIds)
            # Get size of generator matrix
            n, q, batchSize = rG.shape
            p = max(approxErrGenIds) if approxErrGenIds else 0
            if q < p:
                # Append generators for the approximation errors
                rG = np.concatenate([rG, np.zeros((n, len(approxErrGenIds), batchSize), dtype=rG.dtype)], axis=1)
            # Obtain the dn largest approximation errors.
            # [~,dDims] = sort(1/2*(du - dl),1,'descend');
            dDims = np.tile(np.arange(n).reshape(-1, 1), (1, batchSize))
            dDimsIdx = np.ravel_multi_index([dDims, np.tile(np.arange(batchSize), (n, 1))], (n, batchSize))
            notdDimsIdx = dDimsIdx[dn:, :]
            # set not considered approx. error to 0
            dl[notdDimsIdx] = f(c[notdDimsIdx]) - m[notdDimsIdx] * c[notdDimsIdx]
            du[notdDimsIdx] = dl[notdDimsIdx]
            # shift y-intercept by center of approximation errors
            t = 1/2 * (du + dl)
            d = 1/2 * (du - dl)
            
            # Compute indices for approximation errors in the generator
            # matrix.
            GdIdx = np.ravel_multi_index([np.tile(np.arange(n), (1, batchSize)), 
                                        np.reshape(dDims[:dn, :], -1),
                                        np.tile(approxErrGenIds, (1, batchSize)),
                                        np.repeat(np.arange(batchSize), n)], (n, p, batchSize))
            # Store indices of approximation error in generator matrix.
            self.backprop['store']['GdIdx'] = GdIdx
            dDimsIdx = dDimsIdx[:dn, :]
            self.backprop['store']['dDimsIdx'] = dDimsIdx
            # Add approximation errors to the generators.
            rG.flat[GdIdx] = d  # (dDimsIdx);
            
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
                    raise ValueError(f"Unsupported 'options.nn.poly_method': {options['nn']['poly_method']}")
                
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
            t = f(c) - m * c
            # approximation errors are 0
            d = 0
        
        # return coefficients
        coeffs = np.transpose(np.concatenate([m.reshape(m.shape[0], m.shape[1], 1), 
                                            t.reshape(t.shape[0], t.shape[1], 1)], axis=2), (0, 2, 1))
        # Add y-intercept.
        rc = rc + t
        
        return rc, rG, coeffs, d
    
    # Helper methods for CORA functionality
    def _representsa_emptySet(self, obj, eps: float = 1e-10) -> bool:
        """Check if object represents empty set"""
        if EmptySet is not None:
            return hasattr(obj, 'representsa_') and obj.representsa_('emptySet', eps)
        return False
