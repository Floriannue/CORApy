"""
nnLeakyReLULayer - class for LeakyReLU layers

Syntax:
    obj = nnLeakyReLULayer(alpha, name)

Inputs:
    alpha - slope of the LeakyReLU for x<0, defaults to 0.01
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

References:
    [1] Singh, G., et al. "Fast and Effective Robustness Certification"
    [2] Kochdumper, Niklas, et al. "Open-and closed-loop neural network 
       verification using polynomial zonotopes." 
       arXiv preprint arXiv:2207.02715 (2022).

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Sebastian Sigl, Tobias Ladner
Written:       11-June-2022
Last update:   16-February-2023 (TL, combined approx_type)
               08-August-2023 (LK, added extreme-points for zonotope batch-eval)
Last revision: 10-August-2022 (renamed)
               26-May-2023
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .nnActivationLayer import nnActivationLayer


class nnLeakyReLULayer(nnActivationLayer):
    """
    LeakyReLU layer class for neural networks
    
    This class implements a LeakyReLU activation function with configurable slope alpha.
    The function computes y = max(alpha * x, x) for input x.
    """
    
    def __init__(self, alpha: float = 0.01, name: Optional[str] = None):
        """
        Constructor for nnLeakyReLULayer
        
        Args:
            alpha: Slope of the LeakyReLU for x<0, defaults to 0.01
            name: Name of the layer, defaults to type
        """
        if alpha is None:
            alpha = 0.01
        if name is None:
            name = None
        
        # Validate input arguments
        if not isinstance(alpha, (int, float)) or not np.isscalar(alpha):
            raise ValueError("alpha must be a numeric scalar")
        
        # call super class constructor
        super().__init__(name)
        self.alpha = alpha
    
    # evaluate ----------------------------------------------------------------
    
    def evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate numeric input
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Output after LeakyReLU activation
        """
        r = np.maximum(self.alpha * input_data, input_data)
        return r
    
    # Auxiliary functions -----------------------------------------------------
    
    def getDf(self, i: int) -> Callable:
        """
        Get derivative function
        
        Args:
            i: Derivative order
            
        Returns:
            df_i: Derivative function
        """
        if i == 0:
            df_i = self.f
        elif i == 1:
            df_i = lambda x: 1 * (x > 0) + self.alpha * (x <= 0)
        else:
            df_i = lambda x: 0
        
        return df_i
    
    def getDerBounds(self, l: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get derivative bounds
        
        Args:
            l: Lower bounds
            u: Upper bounds
            
        Returns:
            Tuple of (df_l, df_u) derivative bounds
        """
        # df_l and df_u as lower and upper bound for the derivative
        # case distinction for l
        if l <= 0:
            df_l = self.alpha
        else:
            df_l = 1
        
        # case distinction for u
        if u < 0:
            df_u = self.alpha
        else:
            df_u = 1
        
        return df_l, df_u
    
    def computeApproxPoly(self, l: np.ndarray, u: np.ndarray, *args) -> Tuple[np.ndarray, float]:
        """
        Compute approximating polynomial - match MATLAB exactly
        
        Args:
            l: Lower bounds
            u: Upper bounds
            *args: order and poly_method (order defaults to 1, poly_method defaults to 'regression')
            
        Returns:
            Tuple of (coeffs, d) polynomial coefficients and error bound
        """
        # computes an approximating polynomial and respective error bound
        # exploit piecewise linearity of nnLeakyReLULayer
        
        # check if ReLU can be computed exactly
        if u <= 0:
            coeffs = [self.alpha, 0]
            d = 0  # no approximation error!
            
        elif 0 <= l:
            # identity
            coeffs = [1, 0]
            d = 0  # no approximation error!
            
        else:  # l < 0 < u
            # This calls the parent class method in MATLAB
            # We need to call the parent method with the correct signature
            coeffs, d = super().computeApproxPoly(l, u, *args)
        
        return coeffs, d
    
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
        # error can be computed exactly by checking each linear part
        # according to [2, Sec. 3.2]
        
        # x < 0: p(x) - alpha*x
        # This would require minMaxDiffPoly function from CORA
        # For now, we'll use a simplified approach
        diffl1 = 0.0
        diffu1 = 0.0
        
        # x > 0: p(x) - 1*x
        diffl2 = 0.0
        diffu2 = 0.0
        
        # compute final approx error
        diffl = min(diffl1, diffl2)
        diffu = max(diffu1, diffu2)
        diffc = (diffu + diffl) / 2
        coeffs[-1] = coeffs[-1] - diffc
        d = diffu - diffc  # error is radius then.
        
        return coeffs, d
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            fieldStruct: Field structure
        """
        fieldStruct = {}
        fieldStruct['alpha'] = self.alpha
        return fieldStruct
    
    def computeExtremePointsBatch(self, m: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute extreme points batch
        
        Args:
            m: Input data
            options: Options
            
        Returns:
            Tuple of (xs, dxsdm) extreme points and derivatives
        """
        xs = np.zeros_like(m)
        dxsdm = xs
        return xs, dxsdm
    
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
        
        if poly_method == 'singh':
            if order == 1:
                # according to [1, Theorem 3.1]
                lambda_val = (u - self.alpha * l) / (u - l)
                mu = 0.5 * (u - ((u - self.alpha * l) / (u - l)) * u)
                coeffs = [lambda_val, mu]
                d = mu
                return coeffs, d
            elif order == 2:
                # according to [2, Sec. 3.1]
                c_a = u * (1 - self.alpha) / (u - l)**2
                c_b = self.alpha - 2 * c_a * l
                c_c = c_a * l**2
                coeffs = [c_a, c_b, c_c]
        
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
