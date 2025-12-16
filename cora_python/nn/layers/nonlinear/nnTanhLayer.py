"""
nnTanhLayer - class for tanh layers

This class implements a tanh activation layer for neural networks.
"""

import numpy as np
import torch
from typing import Dict, Any
from .nnActivationLayer import nnActivationLayer


class nnTanhLayer(nnActivationLayer):
    """
    Tanh activation layer for neural networks.
    
    The tanh function is defined as: f(x) = tanh(x)
    """
    
    def __init__(self, name=None):
        """
        Constructor for nnTanhLayer.
        
        Args:
            name: name of the layer, defaults to type
        """

        
        # Define the tanh function and its derivative
        def tanh(x):
            return np.tanh(x)
        
        def tanh_derivative(x):
            # Derivative of tanh: f'(x) = 1 - tanh(x)^2
            return 1 - np.tanh(x)**2
        
        def tanh_second_derivative(x):
            # Second derivative of tanh: f''(x) = -2*tanh(x)*(1-tanh(x)^2)
            t = np.tanh(x)
            return -2 * t * (1 - t**2)
        
        super().__init__(name)
        
        # Store derivative functions for higher orders
        self._df2 = tanh_second_derivative
        
        # Pre-computed region polynomials for approximation
        # These are found via obj.findRegionPolys in MATLAB
        self.reg_polys = [
            {'l': -np.inf, 'u': -5, 'p': [0.0000000000000000, -0.9999546021312975], 'd': 0.0000453978687024},
            {'l': -5, 'u': -2.5, 'p': [0.0003632483926582, 0.0077740610313991, 0.0669424281432756, 0.2908011544996281, 0.6397668761140367, -0.4269522964773172], 'd': 0.0000400131387163},
            {'l': -2.5, 'u': -0.625, 'p': [-0.0067465967733122, -0.0683314957855584, -0.2607942809217674, -0.3930263398561141, 0.1220772321097194, 1.1012160065134771, 0.0237533346521373], 'd': 0.0000714657357249},
            {'l': -0.625, 'u': 1.25, 'p': [-0.0055944532555784, -0.0272531518079237, 0.1104113732706064, 0.0164884302761415, -0.3299081428777030, -0.0024990843822790, 0.9998499350506984, 0.0000743310615162], 'd': 0.0000967788202835},
            {'l': 1.25, 'u': 5, 'p': [0.0000512035784496, -0.0014556369685229, 0.0178442098575226, -0.1227508298674645, 0.5143045701658820, -1.3206066384526620, 1.9385133540083290, -0.2654008527398747], 'd': 0.0000712869080566},
            {'l': 5, 'u': np.inf, 'p': [0.0000000000000000, 0.9999546021312975], 'd': 0.0000453978687024}
        ]
    
    def getDf(self, i):
        """
        Get the i-th derivative of the tanh function.
        
        Args:
            i: order of derivative (0, 1, 2)
            
        Returns:
            Function handle for the i-th derivative (torch-compatible)
        """
        if i == 0:
            return self.f
        elif i == 1:
            # Return the tanh derivative function - torch only (internal to nn)
            def tanh_derivative(x):
                # x is already torch (df is internal to nn)
                return 1 - torch.tanh(x)**2
            return tanh_derivative
        elif i == 2:
            # Second derivative - torch only (internal to nn)
            def tanh_second_derivative(x):
                # x is already torch (df is internal to nn)
                t = torch.tanh(x)
                return -2 * t * (1 - t**2)
            return tanh_second_derivative
        else:
            # Higher order derivatives are 0 for tanh - torch only (internal to nn)
            def zero_derivative(x):
                # x is already torch (df is internal to nn)
                return torch.zeros_like(x)
            return zero_derivative
    
    def getDerBounds(self, l, u):
        """
        Get derivative bounds for the tanh function.
        
        Args:
            l: lower bound
            u: upper bound
            
        Returns:
            Tuple of (df_l, df_u) where df_l and df_u are the bounds on the derivative
        """
        # For tanh, the derivative is maximum at x = 0 and decreases as |x| increases
        if l <= 0 <= u:
            # Maximum derivative is at x = 0: f'(0) = 1
            df_max = 1.0
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
        Compute approximation error for the tanh function.
        
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
            
            # Compute extreme points of tanh - mx+t; there are two solutions: xu and xl
            # Limit m to prevent numerical issues
            m = max(min(1, m), np.finfo(float).eps)
            
            if m < 1:
                xu = np.arctanh(np.sqrt(1 - m))  # point with max. upper error
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
        Compute extreme points of tanh - mx+t for batch processing.
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
        
        # Compute extreme points of tanh - mx+t; there are two solutions: xu and xl
        m = torch.maximum(torch.minimum(torch.tensor(1, dtype=dtype, device=device), m), torch.tensor(torch.finfo(dtype).eps, dtype=dtype, device=device))
        
        # Prevent division by zero
        m_safe = torch.maximum(2 * m * torch.sqrt(1 - m), torch.tensor(torch.finfo(dtype).eps, dtype=dtype, device=device))
        
        xu = torch.atanh(torch.sqrt(1 - m))  # point with max. upper error
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
        Evaluate the tanh function numerically.
        Internal to nn - input_data is always torch tensor
        
        Args:
            input_data: input data (torch tensor)
            options: options dictionary
            
        Returns:
            Output of the tanh function (torch tensor)
        """
        # Internal to nn - input_data is always torch tensor
        # Handle numpy arrays (can be passed from aux_imgEncBatch when use_approx_error is True)
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        return torch.tanh(input_data)
