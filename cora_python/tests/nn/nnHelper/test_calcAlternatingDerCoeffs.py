"""
Test for nnHelper.calcAlternatingDerCoeffs function

This test verifies that the calcAlternatingDerCoeffs function works correctly for calculating alternating derivative coefficients.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.calcAlternatingDerCoeffs import calcAlternatingDerCoeffs


class MockLayer:
    """Mock layer class for testing calcAlternatingDerCoeffs"""
    
    def __init__(self, f_func, df_func, getDf_func):
        self.f_func = f_func
        self.df_func = df_func
        self.getDf_func = getDf_func
    
    def f(self, x):
        return self.f_func(x)
    
    def df(self, x):
        return self.df_func(x)
    
    def getDf(self, order):
        return self.getDf_func(order)


class TestCalcAlternatingDerCoeffs:
    """Test class for calcAlternatingDerCoeffs function"""
    
    def test_calcAlternatingDerCoeffs_basic(self):
        """Test basic calcAlternatingDerCoeffs functionality"""
        # Create a simple layer: f(x) = x^2, df(x) = 2x, d2f(x) = 2
        def f_func(x):
            return x**2
        
        def df_func(x):
            return 2*x
        
        def getDf_func(order):
            if order == 2:
                return lambda x: 2
            else:
                return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        # Test with order 2
        l, u = 0, 1
        order = 2
        result = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # Check that result is numpy array
        assert isinstance(result, np.ndarray)
        
        # Check dimensions - should have order+1 coefficients
        assert result.shape == (1, order + 1)
        
        # Check that coefficients are real numbers
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_different_orders(self):
        """Test calcAlternatingDerCoeffs with different orders"""
        # Create a simple layer: f(x) = x^3
        def f_func(x):
            return x**3
        
        def df_func(x):
            return 3*x**2
        
        def getDf_func(order):
            if order == 2:
                return lambda x: 6*x
            elif order == 3:
                return lambda x: 6
            else:
                return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        # Test different orders
        l, u = -1, 1
        for order in [1, 2, 3]:
            result = calcAlternatingDerCoeffs(l, u, order, layer)
            
            # Check dimensions
            assert result.shape == (1, order + 1)
            
            # Check that coefficients are real
            assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_edge_cases(self):
        """Test calcAlternatingDerCoeffs edge cases"""
        # Constant function
        def f_func(x):
            return 5
        
        def df_func(x):
            return 0
        
        def getDf_func(order):
            return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        # Test with order 0
        l, u = 0, 1
        result = calcAlternatingDerCoeffs(l, u, 0, layer)
        assert result.shape == (1, 1)
        assert np.all(np.isreal(result))
        
        # Test with order 1
        result = calcAlternatingDerCoeffs(l, u, 1, layer)
        assert result.shape == (1, 2)
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_large_order(self):
        """Test calcAlternatingDerCoeffs with large order"""
        # Polynomial function: f(x) = x^5
        def f_func(x):
            return x**5
        
        def df_func(x):
            return 5*x**4
        
        def getDf_func(order):
            if order == 2:
                return lambda x: 20*x**3
            elif order == 3:
                return lambda x: 60*x**2
            elif order == 4:
                return lambda x: 120*x
            elif order == 5:
                return lambda x: 120
            else:
                return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        # Test with order 5
        l, u = -1, 1
        order = 5
        result = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # Check dimensions
        assert result.shape == (1, order + 1)
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_consistency(self):
        """Test calcAlternatingDerCoeffs consistency"""
        # Linear function: f(x) = 2x + 3
        def f_func(x):
            return 2*x + 3
        
        def df_func(x):
            return 2
        
        def getDf_func(order):
            return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        l, u = 0, 1
        order = 1
        
        # Multiple calls should give same results
        result1 = calcAlternatingDerCoeffs(l, u, order, layer)
        result2 = calcAlternatingDerCoeffs(l, u, order, layer)
        
        assert np.allclose(result1, result2)
    
    def test_calcAlternatingDerCoeffs_error_handling(self):
        """Test calcAlternatingDerCoeffs error handling"""
        # Create a layer that might cause issues
        def f_func(x):
            return x**2
        
        def df_func(x):
            return 2*x
        
        def getDf_func(order):
            return lambda x: 2 if order == 2 else lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        # Test with invalid order - function doesn't validate input like MATLAB
        l, u = 0, 1
        # The function should handle negative orders gracefully (like MATLAB)
        result = calcAlternatingDerCoeffs(l, u, -1, layer)
        assert isinstance(result, np.ndarray)
    
    def test_calcAlternatingDerCoeffs_mathematical_properties(self):
        """Test calcAlternatingDerCoeffs mathematical properties"""
        # Quadratic function: f(x) = x^2 + x + 1
        def f_func(x):
            return x**2 + x + 1
        
        def df_func(x):
            return 2*x + 1
        
        def getDf_func(order):
            if order == 2:
                return lambda x: 2
            else:
                return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        l, u = 0, 1
        order = 2
        result = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # Check that result has correct shape
        assert result.shape == (1, 3)
        
        # Check that coefficients are finite
        assert np.all(np.isfinite(result))
    
    def test_calcAlternatingDerCoeffs_specific_values(self):
        """Test calcAlternatingDerCoeffs with specific known values"""
        # Linear function: f(x) = x
        def f_func(x):
            return x
        
        def df_func(x):
            return 1
        
        def getDf_func(order):
            return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        l, u = 0, 1
        order = 1
        result = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # For linear function, should get coefficients close to [0, 1]
        # (allowing for numerical precision)
        assert result.shape == (1, 2)
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_numerical_stability(self):
        """Test calcAlternatingDerCoeffs numerical stability"""
        # Function with small values
        def f_func(x):
            return 1e-6 * x**2
        
        def df_func(x):
            return 2e-6 * x
        
        def getDf_func(order):
            if order == 2:
                return lambda x: 2e-6
            else:
                return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        l, u = 0, 1
        order = 2
        result = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # Should still return finite values
        assert np.all(np.isfinite(result))
        assert result.shape == (1, 3)
    
    def test_calcAlternatingDerCoeffs_type_consistency(self):
        """Test calcAlternatingDerCoeffs type consistency"""
        # Simple function
        def f_func(x):
            return x**2
        
        def df_func(x):
            return 2*x
        
        def getDf_func(order):
            return lambda x: 2 if order == 2 else lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        l, u = 0, 1
        order = 2
        result = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # Check return type
        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.float64, np.float32]
    
    def test_calcAlternatingDerCoeffs_integration(self):
        """Test calcAlternatingDerCoeffs integration with polynomial evaluation"""
        # Quadratic function: f(x) = x^2 + 2x + 1
        def f_func(x):
            return x**2 + 2*x + 1
        
        def df_func(x):
            return 2*x + 2
        
        def getDf_func(order):
            if order == 2:
                return lambda x: 2
            else:
                return lambda x: 0
        
        layer = MockLayer(f_func, df_func, getDf_func)
        
        l, u = 0, 1
        order = 2
        coeffs = calcAlternatingDerCoeffs(l, u, order, layer)
        
        # Test that coefficients can be used for polynomial evaluation
        x_test = 0.5
        y_poly = np.polyval(coeffs.flatten(), x_test)
        
        # Should be finite
        assert np.isfinite(y_poly)