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
    
    def __init__(self, name: Optional[str] = None):
        """
        Constructor for nnActivationLayer
        
        Args:
            name: Name of the layer, defaults to type
        """
        # call super class constructor
        super().__init__(name)
        
        # whether the layer is refinable
        self.is_refinable = True
        
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
    
    @abstractmethod
    def getDf(self, i: int) -> Callable:
        """
        Abstract method to get derivative function
        
        Args:
            i: Derivative order
            
        Returns:
            df_i: Derivative function
        """
        pass
    
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
        # This is a placeholder - subclasses should override
        raise NotImplementedError("computeApproxError not implemented in base class")
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            fieldStruct: Field structure
        """
        fieldStruct = {
            'alpha': getattr(self, 'alpha', None)
        }
        return fieldStruct
    
    def getMergeBuckets(self) -> int:
        """
        Get merge buckets for network reduction
        
        Returns:
            buckets: Number of merge buckets
        """
        return 0
    
    def _computeExtremePointsBatch(self, m: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute extreme points batch
        
        Args:
            m: Input data
            options: Options
            
        Returns:
            Tuple of (xs, dxsdm) extreme points and derivatives
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("_computeExtremePointsBatch not implemented in base class")
    
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
        # This is a placeholder - subclasses should override
        raise NotImplementedError("aux_imgEncBatch not implemented in base class")
    
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
        # For now, use the custom method
        order = kwargs.get('order', 1)
        poly_method = kwargs.get('poly_method', 'singh')
        return self._computeApproxPolyCustom(l, u, order, poly_method)
    
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
        # backpropagate gradient
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
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Taylor model
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Taylor model evaluation result
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("evaluateTaylm not implemented in base class")
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id_: List[int], id_2: List[int], 
                            ind: List[int], ind_2: List[int], 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, List[int], List[int], 
                                                           List[int], List[int]]:
        """
        Evaluate polynomial zonotope
        
        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            id_: ID vector
            id_2: ID vector 2
            ind: Index vector
            ind_2: Index vector 2
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("evaluatePolyZonotope not implemented in base class")
    
    def evaluatePolyZonotopeNeuron(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                                  E: np.ndarray, Es: np.ndarray, order: int, 
                                  ind: List[int], ind_: List[int], 
                                  options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Evaluate polynomial zonotope for a specific neuron
        
        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            Es: Exponent matrix 2
            order: Polynomial order
            ind: Index vector
            ind_: Index vector 2
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        # This is a placeholder - subclasses should override
        raise NotImplementedError("evaluatePolyZonotopeNeuron not implemented in base class")
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate constraint zonotope
        
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
        # This is a placeholder - subclasses should override
        raise NotImplementedError("evaluateConZonotope not implemented in base class")
    
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
        # This is a placeholder - subclasses should override
        raise NotImplementedError("evaluateConZonotopeNeuron not implemented in base class")
    
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
        coeffs = self.backprop['store'].get('coeffs', None)
        if coeffs is None:
            raise ValueError("No coefficients stored from forward pass")
        
        # obtain slope of the approximation
        m = coeffs[:, 0, :] if len(coeffs.shape) == 3 else coeffs[:, 0]
        
        # obtain indices of active generators
        genIds = self.backprop['store'].get('genIds', slice(None))
        
        if options.get('nn', {}).get('train', {}).get('exact_backprop', False):
            # This is a complex implementation that would require the full CORA functionality
            # For now, we'll use a simplified version
            gc = gc * m
            gG = gG * m.reshape(-1, 1, 1) if len(m.shape) == 1 else gG * m.reshape(-1, 1, 1)
        else:
            # Consider the approximation as fixed. Use the slope of the approximation for backpropagation
            gc = gc * m
            gG = gG * m.reshape(-1, 1, 1) if len(m.shape) == 1 else gG * m.reshape(-1, 1, 1)
        
        # Clear backprop storage.
        if 'store' in self.backprop:
            self.backprop['store'].clear()
        
        return gc, gG
    
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
        # This is a placeholder - subclasses should override
        raise NotImplementedError("computeApproxError not implemented in base class")
    
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
    
    def _computeApproxPolyCustom(self, l: np.ndarray, u: np.ndarray, order: int, poly_method: str) -> Tuple[np.ndarray, float]:
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
        d = []
        return coeffs, d
    
    def getNumNeurons(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get number of input and output neurons
        
        Returns:
            Tuple of (nin, nout) where each can be None
        """
        # Activation layers don't change the number of neurons
        return None, None
    
    def getOutputSize(self, inputSize: List[int]) -> List[int]:
        """
        Get output size given input size
        
        Args:
            inputSize: Input dimensions
            
        Returns:
            outputSize: Output dimensions
        """
        # Activation layers don't change the output size
        return inputSize

    # Helper methods for CORA functionality
    def _representsa_emptySet(self, obj, eps: float = 1e-10) -> bool:
        """Check if object represents empty set"""
        if EmptySet is not None:
            return hasattr(obj, 'representsa_') and obj.representsa_('emptySet', eps)
        return False

    def evaluateZonotopeBatch(self, c: np.ndarray, G: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate zonotope batch
        
        Args:
            c: Center
            G: Generators
            options: Evaluation options
            
        Returns:
            Tuple of (c, G) results
        """
        # Compute image enclosure
        rc, rG, coeffs = self.aux_imgEncBatch(self.f, self.df, c, G, options, 
                                             lambda m: self._computeExtremePointsBatch(m, options))
        
        # store inputs and coeffs for backpropagation
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            # Store coefficients
            self.backprop['store']['coeffs'] = coeffs
            
            # Store the slope.
            if options.get('nn', {}).get('train', {}).get('exact_backprop', False):
                # Store gradient for the backprop through an image enclosure.
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
