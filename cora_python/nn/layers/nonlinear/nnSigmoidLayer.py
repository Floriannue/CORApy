"""
nnSigmoidLayer - class for Sigmoid layers

This class implements a sigmoid activation layer for neural networks.
"""

import numpy as np
import torch
from typing import Any, Dict
from .nnActivationLayer import nnActivationLayer


class nnSigmoidLayer(nnActivationLayer):
    """
    Sigmoid activation layer for neural networks.
    
    The sigmoid function is defined as: f(x) = 1 / (1 + exp(-x))
    For numerical stability, we use: f(x) = tanh(x/2) / 2 + 0.5
    """
    
    def __init__(self, name=None):
        """
        Constructor for nnSigmoidLayer.
        
        Args:
            name: name of the layer, defaults to type
        """
        
        # Define the sigmoid function and its derivative
        def sigmoid(x):
            # Use tanh for numeric stability
            return np.tanh(x/2) / 2 + 0.5
        
        def sigmoid_derivative(x):
            # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
            s = sigmoid(x)
            return s * (1 - s)
        
        def sigmoid_second_derivative(x):
            # Second derivative of sigmoid
            s = sigmoid(x)
            return s * (1 - s) * (1 - 2*s)
        
        super().__init__(name)
        
        # Store derivative functions for higher orders
        self._df2 = sigmoid_second_derivative
        
        # Pre-computed region polynomials for approximation
        # These are found via obj.findRegionPolys in MATLAB
        self.reg_polys = [
            {'l': -np.inf, 'u': -10, 'p': [0.0000000000000000, 0.0000226989343512], 'd': 0.0000226989343512},
            {'l': -10, 'u': -5, 'p': [0.0000056757561353, 0.0002429394072305, 0.0041839017589345, 0.0363501443122013, 0.1599417190270924, 0.2865242403387572], 'd': 0.0000112339935969},
            {'l': -5, 'u': -1.25, 'p': [-0.0000527077872915, -0.0010676796216488, -0.0081498212788008, -0.0245641462409732, 0.0152596540138349, 0.2753040016287754, 0.5118757437687305], 'd': 0.0000267218430910},
            {'l': -1.25, 'u': 2.5, 'p': [0.0000281300436054, -0.0001624932291702, -0.0000940034934591, 0.0020483035259502, 0.0001012481844055, -0.0208261173225880, -0.0000360364456032, 0.2499994775884087, 0.5000017612516241], 'd': 0.0000093721843051},
            {'l': 2.5, 'u': 10, 'p': [0.0000000076826740, -0.0000001836711225, -0.0000031954035029, 0.0001818834877788, -0.0031382230953991, 0.0290271690503697, -0.1566526667804534, 0.4720559527908412, 0.3752299739623118], 'd': 0.0000130525253081},
            {'l': 10, 'u': np.inf, 'p': [0.0000000000000000, 0.9999773010656487], 'd': 0.0000226989343513}
        ]
    
    def evaluateNumeric(self, input_data: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate numeric input
        Internal to nn - input_data is always torch tensor
        
        Args:
            input_data: Input data (torch tensor)
            options: Evaluation options
            
        Returns:
            r: Output after sigmoid activation (torch tensor)
        """
        # Internal to nn - input_data is always torch tensor
        
        # Use tanh for numeric stability
        r = torch.tanh(input_data/2) / 2 + 0.5
        return r
    
    def getDf(self, i):
        """
        Get the i-th derivative of the sigmoid function.
        
        Args:
            i: order of derivative (0, 1, 2)
            
        Returns:
            Function handle for the i-th derivative (torch-compatible)
        """
        if i == 0:
            return self.f
        elif i == 1:
            # Return the sigmoid derivative function - torch only (internal to nn)
            def sigmoid_derivative(x):
                # x is already torch (df is internal to nn)
                s = torch.tanh(x/2) / 2 + 0.5
                return s * (1 - s)
            return sigmoid_derivative
        elif i == 2:
            # Second derivative - torch only (internal to nn)
            def sigmoid_second_derivative(x):
                # x is already torch (df is internal to nn)
                s = torch.tanh(x/2) / 2 + 0.5
                return s * (1 - s) * (1 - 2*s)
            return sigmoid_second_derivative
        else:
            # Higher order derivatives are 0 for sigmoid - torch only (internal to nn)
            def zero_derivative(x):
                # x is already torch (df is internal to nn)
                return torch.zeros_like(x)
            return zero_derivative
    
    def getDerBounds(self, l, u):
        """
        Get derivative bounds for the sigmoid function.
        
        Args:
            l: lower bound
            u: upper bound
            
        Returns:
            Tuple of (df_l, df_u) where df_l and df_u are the bounds on the derivative
        """
        # For sigmoid, the derivative is maximum at x = 0 and decreases as |x| increases
        if l <= 0 <= u:
            # Maximum derivative is at x = 0: f'(0) = 0.25
            df_max = 0.25
            # Minimum derivative is at the boundary with larger |x|
            if abs(l) > abs(u):
                df_min = self.df(l)
            else:
                df_min = self.df(u)
            return df_min, df_max
        else:
            # Both bounds are on the same side of 0
            df_l = self.df(l)
            df_u = self.df(u)
            return min(df_l, df_u), max(df_l, df_u)
    
    def computeApproxError(self, l, u, coeffs):
        """
        Compute approximation error for the sigmoid function.
        
        Args:
            l: lower bound
            u: upper bound
            coeffs: polynomial coefficients [m, t] for order 1
            
        Returns:
            Tuple of (coeffs, d) where coeffs are the adjusted coefficients and d is the error bound
        """
        order = len(coeffs) - 1
        
        if order == 1:
            # Closed form: see [1] Koller, L. "Co-Design for Training and Verifying Neural Networks"
            m = coeffs[0]
            t = coeffs[1]
            
            # Compute extreme points of sigmoid - mx+t; there are two solutions: xu and xl
            # Limit m to prevent numerical issues
            m = min(1/4, m)
            
            if m > 0:
                xu = 2 * np.arctanh(np.sqrt(1 - 4*m))  # point with max. upper error
                xl = -xu  # point with max. lower error
                
                # Evaluate candidate extreme points within boundary
                xs = np.array([l, xu, xl, u])
                xs = xs[(l <= xs) & (xs <= u)]
                
                if len(xs) > 0:
                    ys = self.f(xs)
                    
                    # Compute approximation error at candidates
                    dBounds = ys - (m * xs + t)
                    
                    # Compute approximation error
                    du = np.max(dBounds)
                    dl = np.min(dBounds)
                    dc = (du + dl) / 2
                    d = du - dc
                    
                    # Shift coeffs by center
                    coeffs = [m, t + dc]
                else:
                    d = 0
            else:
                d = 0
        else:
            # For higher orders, use the default method from parent class
            coeffs, d = super().computeApproxError(l, u, coeffs)
        
        return coeffs, d
    
    def computeExtremePointsBatch(self, m, options):
        """
        Compute extreme points of sigmoid - mx+t for batch processing.
        Works with torch tensors internally.
        
        Args:
            m: slope values (torch tensor expected internally)
            options: options dictionary
            
        Returns:
            Tuple of (xs, dxsdm) where xs are the extreme points and dxsdm are their derivatives (torch tensors)
        """
        # Convert to torch if needed (internal to nn, so should already be torch)
        if isinstance(m, np.ndarray):
            m = torch.tensor(m, dtype=torch.float32)
        
        device = m.device
        dtype = m.dtype
        
        # Compute extreme points of sigmoid - mx+t; there are two solutions: xu and xl
        m = torch.minimum(torch.tensor(1/4, dtype=dtype, device=device), m)
        
        # Prevent division by zero
        m_safe = torch.maximum(m * torch.sqrt(1 - 4*m), torch.tensor(torch.finfo(dtype).eps, dtype=dtype, device=device))
        
        xu = 2 * torch.atanh(torch.sqrt(1 - 4*m))  # point with max. upper error
        xl = -xu  # point with max. lower error
        
        # List of extreme points
        # m has shape (n, 1, b), so xl and xu also have shape (n, 1, b)
        # MATLAB: xs = cat(3,xl,xu); - concatenates along dimension 3
        # Stack along dimension 2 to get (n, 1, 2, b) where dim=2 is the number of extreme points
        xs = torch.stack([xl, xu], dim=2)  # (n, 1, b) -> (n, 1, 2, b)
        
        # Compute derivative wrt. slope m; needed for backpropagation
        dxu = -1 / m_safe
        dxl = -dxu
        dxsdm = torch.stack([dxl, dxu], dim=2)  # (n, 1, b) -> (n, 1, 2, b)
        
        return xs, dxsdm
    
    def evaluateNumeric(self, input_data: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate the sigmoid function numerically.
        Internal to nn - input_data is always torch tensor
        
        Args:
            input_data: input data (torch tensor)
            options: options dictionary
            
        Returns:
            Output of the sigmoid function (torch tensor)
        """
        # Internal to nn - input_data is always torch tensor
        # Handle numpy arrays (can be passed from aux_imgEncBatch when use_approx_error is True)
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        # Use tanh for numeric stability
        return torch.tanh(input_data/2) / 2 + 0.5
