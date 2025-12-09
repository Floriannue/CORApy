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
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .nnActivationLayer import nnActivationLayer
from cora_python.nn.nnHelper.minMaxDiffPoly import minMaxDiffPoly
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


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
        # Validate input arguments
        if not isinstance(alpha, (int, float)) or not np.isscalar(alpha):
            raise ValueError("alpha must be a numeric scalar")
        
        # call super class constructor
        super().__init__(name)
        self.alpha = alpha
    
    # evaluate ----------------------------------------------------------------
    
    def evaluateNumeric(self, input_data, options: Dict[str, Any]):
        """
        Evaluate numeric input
        
        Args:
            input_data: Input data (numpy array or torch tensor) - converted to torch internally
            options: Evaluation options
            
        Returns:
            r: Output after LeakyReLU activation (torch tensor)
        """
        # Convert numpy input to torch if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        r = torch.maximum(self.alpha * input_data, input_data)
        return r
    
    # Auxiliary functions -----------------------------------------------------
    
    def getDf(self, i: int) -> Callable:
        """
        Get derivative function
        
        Args:
            i: Derivative order
            
        Returns:
            df_i: Derivative function (torch-compatible)
        """
        if i == 0:
            df_i = self.f
        elif i == 1:
            def leaky_relu_derivative(x):
                # x is already torch (df is internal to nn)
                return torch.where(x > 0, torch.tensor(1.0, dtype=x.dtype, device=x.device), 
                                 torch.tensor(self.alpha, dtype=x.dtype, device=x.device))
            df_i = leaky_relu_derivative
        else:
            def zero_derivative(x):
                # x is already torch (df is internal to nn)
                return torch.zeros_like(x)
            df_i = zero_derivative
        
        return df_i
    
    def _df2(self, x):
        """
        Second derivative of LeakyReLU (always 0) - torch only (internal to nn)
        
        Args:
            x: Input values (torch tensor)
            
        Returns:
            Second derivative values (all zeros, torch tensor)
        """
        # x is already torch (internal to nn)
        return torch.zeros_like(x)
    
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
        # First validate inputs using parent class validation
        # This ensures consistent validation across all activation layers
        if l > u:
            raise CORAerror('CORA:wrongInputInConstructor', 'l must be <= u')
        
        # computes an approximating polynomial and respective error bound
        # exploit piecewise linearity of nnLeakyReLULayer
        
        # check if ReLU can be computed exactly
        if u <= 0:
            coeffs = [0, self.alpha]  # [constant, linear] - descending order
            d = 0  # no approximation error!
            
        elif 0 <= l:
            # identity
            coeffs = [0, 1]  # [constant, linear] - descending order
            d = 0  # no approximation error!
            
        else:  # l < 0 < u
            # This calls the parent class method in MATLAB
            # We need to call the parent method with the correct signature
            coeffs, d = super().computeApproxPoly(l, u, *args)
            
            # Keep coefficients in descending order (as returned by parent class)
            # This matches MATLAB behavior where polyval expects descending order
        
        return coeffs, d
    
    def computeApproxError(self, l: Union[np.ndarray, torch.Tensor], u: Union[np.ndarray, torch.Tensor], 
                          coeffs: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
        """
        Compute approximation error - works with torch tensors internally
        
        Args:
            l: Lower bounds (torch tensor expected internally)
            u: Upper bounds (torch tensor expected internally)
            coeffs: Polynomial coefficients (torch tensor expected internally)
            
        Returns:
            Tuple of (coeffs, d) updated coefficients and error bound (coeffs as torch tensor)
        """
        # Convert to torch if needed (internal to nn, so should already be torch)
        if isinstance(coeffs, list):
            coeffs = torch.tensor(coeffs, dtype=torch.float32)
        elif isinstance(coeffs, np.ndarray):
            coeffs = torch.tensor(coeffs, dtype=torch.float32)
        
        if isinstance(l, np.ndarray):
            l = torch.tensor(l, dtype=torch.float32)
        if isinstance(u, np.ndarray):
            u = torch.tensor(u, dtype=torch.float32)
        
        # Make a copy to avoid modifying original - use torch clone
        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.clone()
        else:
            coeffs = coeffs.copy()
        
        # Convert l, u to scalars if needed
        l_val = l.item() if isinstance(l, torch.Tensor) and l.numel() == 1 else (float(l) if isinstance(l, (int, float)) else l)
        u_val = u.item() if isinstance(u, torch.Tensor) and u.numel() == 1 else (float(u) if isinstance(u, (int, float)) else u)
        
        # x < 0: p(x) - alpha*x
        # MATLAB: minMaxDiffPoly(coeffs,[obj.alpha,0],l,0)
        # minMaxDiffPoly now handles torch/numpy conversion internally
        # Use torch internally - minMaxDiffPoly will convert if needed
        alpha_array = torch.tensor([self.alpha, 0], dtype=torch.float32)
        diffl1, diffu1 = minMaxDiffPoly(coeffs, alpha_array, l_val, 0)
        
        # x > 0: p(x) - 1*x
        # MATLAB: minMaxDiffPoly(coeffs,[1,0],0,u)
        # Use torch internally - minMaxDiffPoly will convert if needed
        ones_array = torch.tensor([1, 0], dtype=torch.float32)
        diffl2, diffu2 = minMaxDiffPoly(coeffs, ones_array, 0, u_val)
        
        # compute final approx error
        diffl = min(diffl1, diffl2)
        diffu = max(diffu1, diffu2)
        diffc = (diffu + diffl) / 2
        
        # Update coeffs - handle both torch and numpy
        if isinstance(coeffs, torch.Tensor):
            coeffs[-1] = coeffs[-1] - diffc
        else:
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
    
    def computeExtremePointsBatch(self, m: Union[np.ndarray, torch.Tensor], options: Dict[str, Any]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Compute extreme points batch - works with torch tensors internally
        
        Args:
            m: Input data (torch tensor expected internally)
            options: Options
            
        Returns:
            Tuple of (xs, dxsdm) extreme points and derivatives (torch tensors)
        """
        # Convert to torch if needed (internal to nn, so should already be torch)
        if isinstance(m, np.ndarray):
            m = torch.tensor(m, dtype=torch.float32)
        
        # MATLAB: xs = zeros(size(m),'like',m); dxsdm = xs;
        # Use torch internally
        xs = torch.zeros_like(m)
        dxsdm = torch.zeros_like(m)
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
