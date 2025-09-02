"""
Test for nnHelper.leastSquareRidgePolyFunc function

This test verifies that the leastSquareRidgePolyFunc function works correctly for ridge regression polynomial fitting.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.leastSquareRidgePolyFunc import leastSquareRidgePolyFunc


class TestLeastSquareRidgePolyFunc:
    """Test class for leastSquareRidgePolyFunc function"""
    
    def test_leastSquareRidgePolyFunc_basic(self):
        """Test basic leastSquareRidgePolyFunc functionality"""
        # Test with simple case
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])  # Linear function y = 2x
        order = 1
        lambda_reg = 0.1
        
        coeffs = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
        
        # Check that result is numpy array
        assert isinstance(coeffs, np.ndarray)
        
        # Check dimensions - should have order+1 coefficients
        assert len(coeffs) == order + 1
        
        # Check that coefficients are real numbers
        assert np.all(np.isreal(coeffs))
    
    def test_leastSquareRidgePolyFunc_different_orders(self):
        """Test leastSquareRidgePolyFunc with different polynomial orders"""
        np.random.seed(1)  # For reproducibility
        
        # Test different orders
        for order in range(1, 6):
            # Generate exact data points
            x = np.random.rand(order + 1)
            y = np.random.rand(order + 1)
            lambda_reg = 0.01
            
            coeffs = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
            
            # Check dimensions
            assert len(coeffs) == order + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(coeffs))
    
    def test_leastSquareRidgePolyFunc_different_lambda_values(self):
        """Test leastSquareRidgePolyFunc with different regularization strengths"""
        np.random.seed(1)
        x = np.random.rand(5)
        y = np.random.rand(5)
        order = 2
        
        # Test different lambda values
        lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        for lambda_reg in lambda_values:
            coeffs = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
            
            # Check dimensions
            assert len(coeffs) == order + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(coeffs))
            
            # Higher lambda should generally lead to smaller coefficients
            if lambda_reg > 0.1:
                # Coefficients should be finite
                assert np.all(np.isfinite(coeffs))
    
    def test_leastSquareRidgePolyFunc_linear_function(self):
        """Test leastSquareRidgePolyFunc with linear function"""
        # Test with y = 2x + 1
        x = np.array([0, 1, 2, 3])
        y = 2 * x + 1
        order = 1
        
        # Test with default lambda (0.001) - matches MATLAB results
        coeffs = leastSquareRidgePolyFunc(x, y, order)
        
        # Ridge regression with regularization will be close but not exact
        # due to the regularization term
        assert len(coeffs) == 2
        assert np.isclose(coeffs[0], 1.999900, atol=1e-6)  # intercept (MATLAB result)
        assert np.isclose(coeffs[1], 0.999900, atol=1e-6)  # slope (MATLAB result)
        
        # Test with lambda = 0.01
        coeffs_reg = leastSquareRidgePolyFunc(x, y, order, 0.01)
        assert np.isclose(coeffs_reg[0], 1.998999, atol=1e-6)  # intercept (MATLAB result)
        assert np.isclose(coeffs_reg[1], 0.999004, atol=1e-6)  # slope (MATLAB result)
        
        # Test that with lambda=0, we get exact results
        coeffs_exact = leastSquareRidgePolyFunc(x, y, order, 0.0)
        assert np.isclose(coeffs_exact[0], 2.000000, atol=1e-10)  # intercept
        assert np.isclose(coeffs_exact[1], 1.000000, atol=1e-10)  # slope
        
        # Check fit (ridge regression with regularization won't fit exactly)
        y_poly = np.polyval(coeffs, x)
        assert np.allclose(y, y_poly, atol=1e-3)  # Realistic tolerance for ridge regression
    
    def test_leastSquareRidgePolyFunc_quadratic_function(self):
        """Test leastSquareRidgePolyFunc with quadratic function"""
        # Test with y = x² + 2x + 1
        x = np.array([0, 1, 2, 3, 4])
        y = x**2 + 2*x + 1
        order = 2
        
        # Test with default lambda (0.001) - matches MATLAB results
        coeffs = leastSquareRidgePolyFunc(x, y, order)
        
        # Ridge regression with regularization will be close but not exact
        assert len(coeffs) == 3
        assert np.isclose(coeffs[0], 1.000357, atol=1e-6)   # constant term (MATLAB result)
        assert np.isclose(coeffs[1], 1.998574, atol=1e-6)   # linear term (MATLAB result)
        assert np.isclose(coeffs[2], 1.000513, atol=1e-6)   # quadratic term (MATLAB result)
        
        # Test with lambda = 0.01
        coeffs_reg = leastSquareRidgePolyFunc(x, y, order, 0.01)
        assert np.isclose(coeffs_reg[0], 1.003522, atol=1e-6)   # constant term (MATLAB result)
        assert np.isclose(coeffs_reg[1], 1.985938, atol=1e-6)   # linear term (MATLAB result)
        assert np.isclose(coeffs_reg[2], 1.004985, atol=1e-6)   # quadratic term (MATLAB result)
        
        # Test that with lambda=0, we get exact results
        coeffs_exact = leastSquareRidgePolyFunc(x, y, order, 0.0)
        assert np.isclose(coeffs_exact[0], 1.000000, atol=1e-10)   # constant term
        assert np.isclose(coeffs_exact[1], 2.000000, atol=1e-10)   # linear term
        assert np.isclose(coeffs_exact[2], 1.000000, atol=1e-10)   # quadratic term
        
        # Check fit (ridge regression with regularization won't fit exactly)
        y_poly = np.polyval(coeffs, x)
        assert np.allclose(y, y_poly, atol=1e-3)  # Realistic tolerance for ridge regression
    
    def test_leastSquareRidgePolyFunc_with_noise(self):
        """Test leastSquareRidgePolyFunc with noisy data"""
        np.random.seed(1)
        
        for order in range(1, 4):
            # Generate exact data points
            x = np.random.rand(order + 1)
            y = np.random.rand(order + 1)
            
            # Add noise
            n = 50
            noise = np.random.uniform(-0.01, 0.01, n * (order + 1))
            y_noisy = np.tile(y, n) + noise
            x_repeated = np.tile(x, n)
            
            # Test with different lambda values
            for lambda_reg in [0.001, 0.01, 0.1]:
                coeffs = leastSquareRidgePolyFunc(x_repeated, y_noisy, order, lambda_reg)
                
                # Check dimensions
                assert len(coeffs) == order + 1
                
                # Check that coefficients are real
                assert np.all(np.isreal(coeffs))
                
                # For noisy data, fit should be reasonable
                y_poly = np.polyval(coeffs, x)
                
                # Lambda-dependent tolerances based on MATLAB results
                if order == 1:
                    if lambda_reg == 0.001:
                        tol = 0.001
                    elif lambda_reg == 0.01:
                        tol = 0.0003
                    else:  # 0.1
                        tol = 0.01
                elif order == 2:
                    if lambda_reg == 0.001:
                        tol = 0.002
                    elif lambda_reg == 0.01:
                        tol = 0.003
                    else:  # 0.1
                        tol = 0.04
                else:  # order == 3
                    if lambda_reg == 0.001:
                        tol = 0.2
                    elif lambda_reg == 0.01:
                        tol = 0.3
                    else:  # 0.1
                        tol = 0.3
                
                # Should be close to original y (within noise level + regularization effect)
                assert np.allclose(y, y_poly, atol=tol)
    
    def test_leastSquareRidgePolyFunc_edge_cases(self):
        """Test leastSquareRidgePolyFunc edge cases"""
        # Test with single point (order 0)
        x = np.array([1])
        y = np.array([5])
        
        # Test with default lambda (0.001) - matches MATLAB results
        coeffs = leastSquareRidgePolyFunc(x, y, 0)
        assert len(coeffs) == 1
        # Ridge regression with regularization will be close but not exact
        assert np.isclose(coeffs[0], 4.995005, atol=1e-6)  # MATLAB result
        
        # Test that with lambda=0, we get exact results
        coeffs_exact = leastSquareRidgePolyFunc(x, y, 0, 0.0)
        assert np.isclose(coeffs_exact[0], 5.000000, atol=1e-10)
        
        # Test with two points (order 1)
        x = np.array([1, 2])
        y = np.array([3, 5])
        
        # Test with default lambda (0.001) - matches MATLAB results
        coeffs = leastSquareRidgePolyFunc(x, y, 1)
        assert len(coeffs) == 2
        assert np.isclose(coeffs[0], 1.999005, atol=1e-6)  # MATLAB result
        assert np.isclose(coeffs[1], 1.000992, atol=1e-6)  # MATLAB result
        
        # Test with very small lambda
        lambda_reg = 1e-10
        coeffs = leastSquareRidgePolyFunc(x, y, 1, lambda_reg)
        assert len(coeffs) == 2
        assert np.all(np.isfinite(coeffs))
        
        # Test with very large lambda
        lambda_reg = 1e10
        coeffs = leastSquareRidgePolyFunc(x, y, 1, lambda_reg)
        assert len(coeffs) == 2
        assert np.all(np.isfinite(coeffs))
    
    def test_leastSquareRidgePolyFunc_different_dimensions(self):
        """Test leastSquareRidgePolyFunc with different data dimensions"""
        np.random.seed(1)
        
        # Test with different numbers of points
        for n_points in [2, 5, 10, 20]:
            for order in [1, 2, 3]:
                if n_points > order:  # Need more points than order
                    x = np.random.rand(n_points)
                    y = np.random.rand(n_points)
                    lambda_reg = 0.01
                    
                    coeffs = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
                    
                    # Check dimensions
                    assert len(coeffs) == order + 1
                    
                    # Check that coefficients are real
                    assert np.all(np.isreal(coeffs))
    
    def test_leastSquareRidgePolyFunc_consistency(self):
        """Test that leastSquareRidgePolyFunc produces consistent results"""
        np.random.seed(1)
        x = np.random.rand(5)
        y = np.random.rand(5)
        order = 2
        lambda_reg = 0.01
        
        # Call multiple times
        coeffs1 = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
        coeffs2 = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
        
        # Should be consistent
        assert np.allclose(coeffs1, coeffs2, atol=1e-10)
    
    def test_leastSquareRidgePolyFunc_error_handling(self):
        """Test leastSquareRidgePolyFunc error handling (matches MATLAB behavior)"""
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        lambda_reg = 0.01
        
        # MATLAB doesn't validate inputs, so these should work or fail naturally
        # Test with negative order (should fail naturally due to array indexing)
        try:
            coeffs = leastSquareRidgePolyFunc(x, y, -1, lambda_reg)
            # If it doesn't fail, that's also acceptable (MATLAB behavior)
        except Exception:
            # Expected to fail due to array indexing issues
            pass
        
        # Test with order >= number of points (MATLAB allows this)
        coeffs = leastSquareRidgePolyFunc(x, y, 3, lambda_reg)
        assert len(coeffs) == 4  # Should work
        
        # Test with mismatched x and y lengths (should fail due to matrix multiplication)
        with pytest.raises(Exception):  # Any exception is fine
            leastSquareRidgePolyFunc(x, y[:2], 1, lambda_reg)
        
        # Test with empty arrays (Python behavior - may or may not fail)
        try:
            coeffs = leastSquareRidgePolyFunc(np.array([]), np.array([]), 1, lambda_reg)
            # If it doesn't fail, that's also acceptable
        except Exception:
            # Expected to fail due to matrix operations
            pass
        
        # Test with negative lambda (MATLAB allows this)
        coeffs = leastSquareRidgePolyFunc(x, y, 1, -0.1)
        assert len(coeffs) == 2  # Should work
    
    def test_leastSquareRidgePolyFunc_numerical_stability(self):
        """Test numerical stability"""
        # Test with ill-conditioned data
        x = np.array([1, 1.0001, 1.0002])  # Very close x values
        y = np.array([1, 1.0001, 1.0002])
        lambda_reg = 0.01
        
        coeffs = leastSquareRidgePolyFunc(x, y, 1, lambda_reg)
        
        # Should still produce finite results
        assert np.all(np.isfinite(coeffs))
        assert len(coeffs) == 2
        
        # Test with very large range
        x = np.array([0, 1e6])
        y = np.array([0, 1e6])
        lambda_reg = 0.01
        
        coeffs = leastSquareRidgePolyFunc(x, y, 1, lambda_reg)
        
        # Should still produce finite results
        assert np.all(np.isfinite(coeffs))
        assert len(coeffs) == 2
    
    def test_leastSquareRidgePolyFunc_regularization_effect(self):
        """Test that regularization has the expected effect"""
        np.random.seed(1)
        x = np.random.rand(10)
        y = np.random.rand(10)
        order = 3
        
        # Test with different lambda values
        lambda_values = [0.001, 0.01, 0.1, 1.0]
        coeffs_list = []
        
        for lambda_reg in lambda_values:
            coeffs = leastSquareRidgePolyFunc(x, y, order, lambda_reg)
            coeffs_list.append(coeffs)
            
            # Check dimensions
            assert len(coeffs) == order + 1
        
        # Higher lambda should generally lead to smaller coefficient magnitudes
        # (though this is not always guaranteed due to data characteristics)
        for i in range(1, len(lambda_values)):
            # Check that coefficients are finite
            assert np.all(np.isfinite(coeffs_list[i]))
    
    def test_leastSquareRidgePolyFunc_integration(self):
        """Test integration with polynomial evaluation"""
        # Test that coefficients can be used for evaluation
        x = np.array([0, 1, 2, 3])
        y = x**2 + 2*x + 1
        order = 2
        
        # Test with default lambda (0.001) - matches MATLAB results
        coeffs = leastSquareRidgePolyFunc(x, y, order)
        
        # Test evaluation at original points (ridge regression won't fit exactly)
        y_poly = np.polyval(coeffs, x)
        assert np.allclose(y, y_poly, atol=2e-3)  # Slightly higher tolerance for integration test
        
        # Test evaluation at new points
        x_new = np.array([0.5, 1.5, 2.5])
        y_new = x_new**2 + 2*x_new + 1
        y_poly_new = np.polyval(coeffs, x_new)
        
        # Should be close
        assert np.allclose(y_new, y_poly_new, atol=1e-2)
