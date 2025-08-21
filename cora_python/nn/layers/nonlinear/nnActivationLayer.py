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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from cora_python.nn.nnHelper import leastSquarePolyFunc
from cora_python.nn.nnHelper import leastSquareRidgePolyFunc
from cora_python.nn.nnHelper import minMaxDiffOrder

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
        
        # initialize backprop storage
        self.backprop = {'store': {}}
    
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
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, 
                            id: np.ndarray, id_: np.ndarray, ind: np.ndarray, ind_: np.ndarray, 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate polyZonotope input
        
        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            id: Identifiers
            id_: Identifiers
            ind: Indices
            ind_: Indices
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        # This method needs to be implemented based on MATLAB logic
        # For now, return placeholders
        raise CORAerror('CORA:notSupported', 'evaluatePolyZonotope not implemented yet')
    
    def evaluatePolyZonotopeNeuron(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, 
                                  Es: np.ndarray, order: int, ind: np.ndarray, ind_: np.ndarray, 
                                  options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Evaluate polyZonotope for a specific neuron
        
        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            Es: Exponent matrix
            order: Polynomial order
            ind: Indices
            ind_: Indices
            options: Evaluation options
            
        Returns:
            Tuple of (c, G, GI, d) results
        """
        # This method needs to be implemented based on MATLAB logic
        # For now, return placeholders
        raise CORAerror('CORA:notSupported', 'evaluatePolyZonotopeNeuron not implemented yet')
    
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
        # This method needs to be implemented based on MATLAB logic
        # For now, return placeholder
        raise CORAerror('CORA:notSupported', 'evaluateTaylm not implemented yet')
    
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
        # This is an abstract method that subclasses must implement
        # In MATLAB, this calls nnHelper.lookupDf(obj,i)
        raise CORAerror('CORA:notDefined', 'getDf must be implemented in subclasses')
    
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
            raise CORAerror('CORA:notSupported', f"Unsupported 'options.nn.poly_method': {options['nn']['poly_method']}")
        
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
            t = f(c) - m * c
            # approximation errors are 0
            d = 0
        
        # return coefficients
        coeffs = np.transpose(np.concatenate([m.reshape(m.shape[0], m.shape[1], 1), 
                                            t.reshape(t.shape[0], t.shape[1], 1)], axis=2), (0, 2, 1))
        # Add y-intercept.
        rc = rc + t
        
        return rc, rG, coeffs, d
    


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
