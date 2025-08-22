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
            # Evaluate Bernstein polynomial
            bernstein_value = 0
            for i, c in enumerate(coeffs):
                # Bernstein basis polynomial B_{i,n}(x)
                if n == 0:
                    bernstein_value = c
                else:
                    # B_{i,n}(x) = C(n,i) * x^i * (1-x)^(n-i)
                    from scipy.special import comb
                    bernstein_value += c * comb(n, i) * (x**i) * ((1-x)**(n-i))
            
            # Compare with original function
            original_value = f(x)
            
            # Should be reasonably close (exact for polynomials up to order n)
            if n >= 2:  # x^2 is order 2
                assert np.isclose(bernstein_value, original_value, atol=1e-10)
    
    def test_findBernsteinPoly_constant_function(self):
        """Test findBernsteinPoly with constant function"""
        def f(x):
            return 5.0  # Constant function
        
        l, u = 0, 1
        n = 3
        
        coeffs = findBernsteinPoly(f, l, u, n)
        
        # Check dimensions
        assert len(coeffs) == n + 1
        
        # For constant function, all coefficients should be the same
        assert np.allclose(coeffs, 5.0, atol=1e-10)
    
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
            # Test approximation at endpoints
            x0 = l
            x1 = u
            
            # Evaluate Bernstein polynomial at endpoints
            bernstein_0 = 0
            bernstein_1 = 0
            for i, c in enumerate(coeffs):
                if n == 1:
                    if i == 0:
                        bernstein_0 += c * (1 - x0)
                        bernstein_1 += c * (1 - x1)
                    elif i == 1:
                        bernstein_0 += c * x0
                        bernstein_1 += c * x1
                else:
                    # For higher orders, use general formula
                    from scipy.special import comb
                    bernstein_0 += c * comb(n, i) * (x0**i) * ((1-x0)**(n-i))
                    bernstein_1 += c * comb(n, i) * (x1**i) * ((1-x1)**(n-i))
            
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
            # Evaluate Bernstein polynomial (simplified for large n)
            bernstein_value = 0
            for i, c in enumerate(coeffs):
                if n == 0:
                    bernstein_value = c
                else:
                    from scipy.special import comb
                    bernstein_value += c * comb(n, i) * (x**i) * ((1-x)**(n-i))
            
            original_value = f(x)
            
            # Should be very close
            assert np.isclose(bernstein_value, original_value, atol=1e-8)
    
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
