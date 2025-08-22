"""
Test for nnHelper.leastSquarePolyFunc function

This test verifies that the leastSquarePolyFunc function works correctly for least squares polynomial fitting.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc


class TestLeastSquarePolyFunc:
    """Test class for leastSquarePolyFunc function"""
    
    def test_leastSquarePolyFunc_basic(self):
        """Test basic leastSquarePolyFunc functionality"""
        # Test with simple case
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])  # Linear function y = 2x
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        
        # Check that result is numpy array
        assert isinstance(coeffs, np.ndarray)
        
        # Check dimensions - should have order+1 coefficients
        assert len(coeffs) == 2  # order 1 -> 2 coefficients
        
        # Check that coefficients are real numbers
        assert np.all(np.isreal(coeffs))
    
    def test_leastSquarePolyFunc_different_orders(self):
        """Test leastSquarePolyFunc with different polynomial orders"""
        np.random.seed(1)  # For reproducibility
        
        # Test different orders
        for order in range(1, 6):
            # Generate exact data points
            x = np.random.rand(order + 1)
            y = np.random.rand(order + 1)
            
            coeffs = leastSquarePolyFunc(x, y, order)
            
            # Check dimensions
            assert len(coeffs) == order + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(coeffs))
            
            # For exact fit, polynomial should pass through all points
            y_poly = np.polyval(coeffs, x)
            assert np.allclose(y, y_poly, atol=1e-10)
    
    def test_leastSquarePolyFunc_with_noise(self):
        """Test leastSquarePolyFunc with noisy data"""
        np.random.seed(1)
        
        for order in range(1, 6):
            # Generate exact data points
            x = np.random.rand(order + 1)
            y = np.random.rand(order + 1)
            
            # Add noise
            n = 50
            noise = np.random.uniform(-0.0001, 0.0001, n * (order + 1))
            y_noisy = np.tile(y, n) + noise
            x_repeated = np.tile(x, n)
            
            # Compute coefficients
            coeffs = leastSquarePolyFunc(x_repeated, y_noisy, order)
            
            # Check dimensions
            assert len(coeffs) == order + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(coeffs))
            
            # For noisy data, fit should be reasonable but not exact
            y_poly = np.polyval(coeffs, x)
            # Should be close to original y (within noise level)
            assert np.allclose(y, y_poly, atol=1e-3)
    
    def test_leastSquarePolyFunc_linear_function(self):
        """Test leastSquarePolyFunc with linear function"""
        # Test with y = 2x + 1
        x = np.array([0, 1, 2, 3])
        y = 2 * x + 1
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        
        # Should get coefficients close to [1, 2] (intercept, slope)
        assert len(coeffs) == 2
        assert np.isclose(coeffs[0], 1, atol=1e-10)  # intercept
        assert np.isclose(coeffs[1], 2, atol=1e-10)  # slope
        
        # Check fit
        y_poly = np.polyval(coeffs, x)
        assert np.allclose(y, y_poly, atol=1e-10)
    
    def test_leastSquarePolyFunc_quadratic_function(self):
        """Test leastSquarePolyFunc with quadratic function"""
        # Test with y = x² + 2x + 1
        x = np.array([0, 1, 2, 3, 4])
        y = x**2 + 2*x + 1
        
        coeffs = leastSquarePolyFunc(x, y, 2)
        
        # Should get coefficients close to [1, 2, 1]
        assert len(coeffs) == 3
        assert np.isclose(coeffs[0], 1, atol=1e-10)   # constant term
        assert np.isclose(coeffs[1], 2, atol=1e-10)   # linear term
        assert np.isclose(coeffs[2], 1, atol=1e-10)   # quadratic term
        
        # Check fit
        y_poly = np.polyval(coeffs, x)
        assert np.allclose(y, y_poly, atol=1e-10)
    
    def test_leastSquarePolyFunc_edge_cases(self):
        """Test leastSquarePolyFunc edge cases"""
        # Test with single point (order 0)
        x = np.array([1])
        y = np.array([5])
        
        coeffs = leastSquarePolyFunc(x, y, 0)
        assert len(coeffs) == 1
        assert np.isclose(coeffs[0], 5, atol=1e-10)
        
        # Test with two points (order 1)
        x = np.array([1, 2])
        y = np.array([3, 5])
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        assert len(coeffs) == 2
        
        # Test with very small values
        x = np.array([1e-10, 2e-10])
        y = np.array([1e-10, 2e-10])
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        assert len(coeffs) == 2
        assert np.all(np.isfinite(coeffs))
        
        # Test with very large values
        x = np.array([1e10, 2e10])
        y = np.array([1e10, 2e10])
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        assert len(coeffs) == 2
        assert np.all(np.isfinite(coeffs))
    
    def test_leastSquarePolyFunc_different_dimensions(self):
        """Test leastSquarePolyFunc with different data dimensions"""
        np.random.seed(1)
        
        # Test with different numbers of points
        for n_points in [2, 5, 10, 20]:
            for order in [1, 2, 3]:
                if n_points > order:  # Need more points than order
                    x = np.random.rand(n_points)
                    y = np.random.rand(n_points)
                    
                    coeffs = leastSquarePolyFunc(x, y, order)
                    
                    # Check dimensions
                    assert len(coeffs) == order + 1
                    
                    # Check that coefficients are real
                    assert np.all(np.isreal(coeffs))
    
    def test_leastSquarePolyFunc_consistency(self):
        """Test that leastSquarePolyFunc produces consistent results"""
        np.random.seed(1)
        x = np.random.rand(5)
        y = np.random.rand(5)
        order = 2
        
        # Call multiple times
        coeffs1 = leastSquarePolyFunc(x, y, order)
        coeffs2 = leastSquarePolyFunc(x, y, order)
        
        # Should be consistent
        assert np.allclose(coeffs1, coeffs2, atol=1e-10)
    
    def test_leastSquarePolyFunc_error_handling(self):
        """Test leastSquarePolyFunc error handling"""
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        
        # Test with negative order
        with pytest.raises(ValueError):
            leastSquarePolyFunc(x, y, -1)
        
        # Test with order >= number of points
        with pytest.raises(ValueError):
            leastSquarePolyFunc(x, y, 3)  # 3 points, order 3
        
        # Test with mismatched x and y lengths
        with pytest.raises(ValueError):
            leastSquarePolyFunc(x, y[:2], 1)
        
        # Test with empty arrays
        with pytest.raises(ValueError):
            leastSquarePolyFunc(np.array([]), np.array([]), 1)
    
    def test_leastSquarePolyFunc_numerical_stability(self):
        """Test numerical stability"""
        # Test with ill-conditioned data
        x = np.array([1, 1.0001, 1.0002])  # Very close x values
        y = np.array([1, 1.0001, 1.0002])
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        
        # Should still produce finite results
        assert np.all(np.isfinite(coeffs))
        assert len(coeffs) == 2
        
        # Test with very large range
        x = np.array([0, 1e6])
        y = np.array([0, 1e6])
        
        coeffs = leastSquarePolyFunc(x, y, 1)
        
        # Should still produce finite results
        assert np.all(np.isfinite(coeffs))
        assert len(coeffs) == 2
    
    def test_leastSquarePolyFunc_integration(self):
        """Test integration with polynomial evaluation"""
        # Test that coefficients can be used for evaluation
        x = np.array([0, 1, 2, 3])
        y = x**2 + 2*x + 1
        
        coeffs = leastSquarePolyFunc(x, y, 2)
        
        # Test evaluation at original points
        y_poly = np.polyval(coeffs, x)
        assert np.allclose(y, y_poly, atol=1e-10)
        
        # Test evaluation at new points
        x_new = np.array([0.5, 1.5, 2.5])
        y_new = x_new**2 + 2*x_new + 1
        y_poly_new = np.polyval(coeffs, x_new)
        
        # Should be close
        assert np.allclose(y_new, y_poly_new, atol=1e-10)
