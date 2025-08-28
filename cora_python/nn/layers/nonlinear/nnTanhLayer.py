"""
nnTanhLayer - class for tanh layers

This class implements a tanh activation layer for neural networks.
"""

import numpy as np
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
        if name is None:
            name = "tanh"
        
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
            Function handle for the i-th derivative
        """
        if i == 0:
            return self.f
        elif i == 1:
            # Return the tanh derivative function directly
            def tanh_derivative(x):
                return 1 - np.tanh(x)**2
            return tanh_derivative
        elif i == 2:
            return self._df2
        else:
            # Higher order derivatives are 0 for tanh
            return lambda x: np.zeros_like(x)
    
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
        
        Args:
            m: slope values
            options: options dictionary
            
        Returns:
            Tuple of (xs, dxsdm) where xs are the extreme points and dxsdm are their derivatives
        """
        # Compute extreme points of tanh - mx+t; there are two solutions: xu and xl
        m = np.maximum(np.minimum(1, m), np.finfo(m.dtype).eps)
        
        # Prevent division by zero
        m_safe = np.maximum(2 * m * np.sqrt(1 - m), np.finfo(m.dtype).eps)
        
        xu = np.arctanh(np.sqrt(1 - m))  # point with max. upper error
        xl = -xu  # point with max. lower error
        
        # List of extreme points
        xs = np.stack([xl, xu], axis=2)
        
        # Compute derivative wrt. slope m; needed for backpropagation
        dxu = -1 / m_safe
        dxl = -dxu
        dxsdm = np.stack([dxl, dxu], axis=2)
        
        return xs, dxsdm
    
    def evaluateNumeric(self, input_data, options):
        """
        Evaluate the tanh function numerically.
        
        Args:
            input_data: input data
            options: options dictionary
            
        Returns:
            Output of the tanh function
        """
        return np.tanh(input_data)
