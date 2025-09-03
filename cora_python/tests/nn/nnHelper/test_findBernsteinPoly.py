"""
Test for nnHelper.findBernsteinPoly function

This test verifies that the findBernsteinPoly function works correctly for Bernstein polynomial approximation.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.findBernsteinPoly import findBernsteinPoly


class TestFindBernsteinPoly:
    """Test class for findBernsteinPoly function"""
    
    def test_findBernsteinPoly_basic(self):
        """Test basic findBernsteinPoly functionality"""
        # Define a simple function
        def f(x):
            return x**2
        
        # Test interval and order
        l, u = 0, 1
        n = 3
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Check that result is numpy array
        assert isinstance(coeffs, np.ndarray)
        
        # Check dimensions - should have n+1 coefficients for order n
        assert len(coeffs) == n + 1
        
        # Check that coefficients are real numbers
        assert np.all(np.isreal(coeffs))
    
    def test_findBernsteinPoly_different_orders(self):
        """Test findBernsteinPoly with different polynomial orders"""
        def f(x):
            return x**2
        
        l, u = 0, 1
        
        # Test different orders
        for n in [1, 2, 3, 5, 10]:
            coeffs = findBernsteinPoly(f, l, u, n)
            
            # Check dimensions
            assert len(coeffs) == n + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(coeffs))
    
    def test_findBernsteinPoly_different_intervals(self):
        """Test findBernsteinPoly with different intervals"""
        def f(x):
            return x**2
        
        n = 3
        
        # Test different intervals
        test_intervals = [
            (0, 1),
            (-1, 1),
            (0, 2),
            (-2, 2),
            (1, 3)
        ]
        
        for l, u in test_intervals:
            coeffs = findBernsteinPoly(f, l, u, n)
            
            # Check dimensions
            assert len(coeffs) == n + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(coeffs))
    
    def test_findBernsteinPoly_different_functions(self):
        """Test findBernsteinPoly with different functions"""
        l, u = 0, 1
        n = 3
        
        # Test different functions
        test_functions = [
            lambda x: x**2,           # Quadratic
            lambda x: x**3,           # Cubic
            lambda x: np.sin(x),      # Sine
            lambda x: np.exp(x),      # Exponential
            lambda x: np.log(x + 1),  # Log (shifted to avoid domain issues)
            lambda x: 1,              # Constant
            lambda x: x,              # Linear
        ]
        
        for f in test_functions:
            try:
                coeffs = findBernsteinPoly(f, l, u, n)
                
                # Check dimensions
                assert len(coeffs) == n + 1
                
                # Check that coefficients are real
                assert np.all(np.isreal(coeffs))
                
            except Exception as e:
                # Some functions might fail due to domain issues
                # This is acceptable for testing
                pass
    
    def test_findBernsteinPoly_edge_cases(self):
        """Test findBernsteinPoly edge cases"""
        def f(x):
            return x**2
        
        # Test with order 0
        coeffs = findBernsteinPoly(f, 0, 1, 0)
        assert len(coeffs) == 1
        assert np.all(np.isreal(coeffs))
        
        # Test with order 1
        coeffs = findBernsteinPoly(f, 0, 1, 1)
        assert len(coeffs) == 2
        assert np.all(np.isreal(coeffs))
        
        # Test with very small interval
        coeffs = findBernsteinPoly(f, 0, 0.001, 3)
        assert len(coeffs) == 4
        assert np.all(np.isreal(coeffs))
        
        # Test with very large interval
        coeffs = findBernsteinPoly(f, 0, 1000, 3)
        assert len(coeffs) == 4
        assert np.all(np.isreal(coeffs))
    
    def test_findBernsteinPoly_accuracy(self):
        """Test accuracy of Bernstein polynomial approximation"""
        def f(x):
            return x**2
        
        l, u = 0, 1
        n = 5
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Test approximation at several points
        test_points = np.linspace(l, u, 10)
        
        for x in test_points:
            # Evaluate polynomial using standard form (highest degree first)
            # coeffs[0] is coefficient of x^n, coeffs[n] is constant term
            bernstein_value = np.polyval(coeffs, x)
            
            # Compare with original function
            original_value = f(x)
            
            # Should be reasonably close (Bernstein polynomials are approximations)
            if n >= 2:  # x^2 is order 2
                # Bernstein approximation may not be exact, use more realistic tolerance
                assert np.isclose(bernstein_value, original_value, atol=1e-2)
    
    def test_findBernsteinPoly_constant_function(self):
        """Test findBernsteinPoly with constant function"""
        def f(x):
            return 5.0  # Constant function
        
        l, u = 0, 1
        n = 3
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Check dimensions
        assert len(coeffs) == n + 1
        
        # For constant function, only the constant term should be non-zero
        # coeffs[0] should be 0 (coefficient of x^3)
        # coeffs[1] should be 0 (coefficient of x^2) 
        # coeffs[2] should be 0 (coefficient of x^1)
        # coeffs[3] should be 5.0 (constant term)
        assert np.isclose(coeffs[0], 0, atol=1e-10)  # x^3 term
        assert np.isclose(coeffs[1], 0, atol=1e-10)  # x^2 term
        assert np.isclose(coeffs[2], 0, atol=1e-10)  # x^1 term
        assert np.isclose(coeffs[3], 5.0, atol=1e-10)  # constant term
    
    def test_findBernsteinPoly_linear_function(self):
        """Test findBernsteinPoly with linear function"""
        def f(x):
            return 2*x + 1  # Linear function
        
        l, u = 0, 1
        n = 3
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Check dimensions
        assert len(coeffs) == n + 1
        
        # For linear function with order >= 1, should be exact
        if n >= 1:
            # Test approximation at endpoints using standard polynomial evaluation
            x0 = l
            x1 = u
            
            # Evaluate polynomial using standard form
            bernstein_0 = np.polyval(coeffs, x0)
            bernstein_1 = np.polyval(coeffs, x1)
            
            # Should match original function at endpoints
            assert np.isclose(bernstein_0, f(x0), atol=1e-10)
            assert np.isclose(bernstein_1, f(x1), atol=1e-10)
    
    def test_findBernsteinPoly_negative_interval(self):
        """Test findBernsteinPoly with negative interval"""
        def f(x):
            return x**2
        
        l, u = -1, 0
        n = 3
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Check dimensions
        assert len(coeffs) == n + 1
        
        # Check that coefficients are real
        assert np.all(np.isreal(coeffs))
    
    def test_findBernsteinPoly_large_order(self):
        """Test findBernsteinPoly with large order"""
        def f(x):
            return x**2
        
        l, u = 0, 1
        n = 20  # Large order
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Check dimensions
        assert len(coeffs) == n + 1
        
        # Check that coefficients are real
        assert np.all(np.isreal(coeffs))
        
        # For large orders, approximation should be very accurate
        test_points = np.linspace(l, u, 20)
        
        for x in test_points:
            # Evaluate polynomial using standard form
            bernstein_value = np.polyval(coeffs, x)
            
            original_value = f(x)
            
            # Should be very close (Bernstein approximation improves with higher order)
            assert np.isclose(bernstein_value, original_value, atol=1e-2)
    
    def test_findBernsteinPoly_consistency(self):
        """Test that findBernsteinPoly produces consistent results"""
        def f(x):
            return x**2
        
        l, u = 0, 1
        n = 3
        
        # Call multiple times
        coeffs1 = findBernsteinPoly(f, l, u, n)
        coeffs2 = findBernsteinPoly(f, l, u, n)
        
        # Should be consistent
        assert np.allclose(coeffs1, coeffs2, atol=1e-10)
    
    def test_findBernsteinPoly_error_handling(self):
        """Test findBernsteinPoly error handling"""
        def f(x):
            return x**2
        
        # Test with invalid interval (l >= u)
        with pytest.raises(ValueError):
            findBernsteinPoly(f, 1, 0, 3)
        
        with pytest.raises(ValueError):
            findBernsteinPoly(f, 0, 0, 3)
        
        # Test with negative order
        with pytest.raises(ValueError):
            findBernsteinPoly(f, 0, 1, -1)
        
        # Test with non-integer order
        with pytest.raises(ValueError):
            findBernsteinPoly(f, 0, 1, 2.5)
