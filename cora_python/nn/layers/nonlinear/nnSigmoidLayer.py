"""
nnSigmoidLayer - class for Sigmoid layers

This class implements a sigmoid activation layer for neural networks.
"""

import numpy as np
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
        if name is None:
            name = "sigmoid"
        
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
        
        super().__init__(sigmoid, sigmoid_derivative, name)
        
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
    
    def getDf(self, i):
        """
        Get the i-th derivative of the sigmoid function.
        
        Args:
            i: order of derivative (0, 1, 2)
            
        Returns:
            Function handle for the i-th derivative
        """
        if i == 0:
            return self.f
        elif i == 1:
            return self.df
        elif i == 2:
            return self._df2
        else:
            # Higher order derivatives are 0 for sigmoid
            return lambda x: np.zeros_like(x)
    
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
        
        Args:
            m: slope values
            options: options dictionary
            
        Returns:
            Tuple of (xs, dxsdm) where xs are the extreme points and dxsdm are their derivatives
        """
        # Compute extreme points of sigmoid - mx+t; there are two solutions: xu and xl
        m = np.minimum(1/4, m)
        
        # Prevent division by zero
        m_safe = np.maximum(m * np.sqrt(1 - 4*m), np.finfo(m.dtype).eps)
        
        xu = 2 * np.arctanh(np.sqrt(1 - 4*m))  # point with max. upper error
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
        Evaluate the sigmoid function numerically.
        
        Args:
            input_data: input data
            options: options dictionary
            
        Returns:
            Output of the sigmoid function
        """
        # Use tanh for numeric stability
        return np.tanh(input_data/2) / 2 + 0.5
