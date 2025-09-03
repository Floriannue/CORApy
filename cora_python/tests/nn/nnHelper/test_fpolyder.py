"""
Test for nnHelper.fpolyder function

This test verifies that the fpolyder function works correctly for polynomial differentiation.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.fpolyder import fpolyder


class TestFpolyder:
    """Test class for fpolyder function"""
    
    def test_fpolyder_basic(self):
        """Test basic fpolyder functionality"""
        # Test with simple polynomial coefficients
        coeffs = np.array([1, 2, 3])  # 1 + 2x + 3x²
        
        result = fpolyder(coeffs)
        
        # Check that result is numpy array
        assert isinstance(result, np.ndarray)
        
        # Check dimensions - should have one fewer coefficient
        assert len(result) == len(coeffs) - 1
        
        # Check that coefficients are real numbers
        assert np.all(np.isreal(result))
    
    def test_fpolyder_different_orders(self):
        """Test fpolyder with different polynomial orders"""
        # Test order 1 (linear) - MATLAB: [1, 2] represents 2 + 1*x
        coeffs1 = np.array([1, 2])  # 2 + 1*x
        result1 = fpolyder(coeffs1)
        assert len(result1) == 1
        assert np.isclose(result1[0], 1, atol=1e-10)  # derivative of 2 + 1*x is 1
        
        # Test order 2 (quadratic) - MATLAB: [1, 2, 3] represents 3 + 2*x + 1*x²
        coeffs2 = np.array([1, 2, 3])  # 3 + 2*x + 1*x²
        result2 = fpolyder(coeffs2)
        assert len(result2) == 2
        assert np.isclose(result2[0], 2, atol=1e-10)      # derivative of 2*x is 2
        assert np.isclose(result2[1], 2, atol=1e-10)      # derivative of 1*x² is 2*x
        
        # Test order 3 (cubic) - MATLAB: [1, 2, 3, 4] represents 4 + 3*x + 2*x² + 1*x³
        coeffs3 = np.array([1, 2, 3, 4])  # 4 + 3*x + 2*x² + 1*x³
        result3 = fpolyder(coeffs3)
        assert len(result3) == 3
        assert np.isclose(result3[0], 3, atol=1e-10)      # derivative of 3*x is 3
        assert np.isclose(result3[1], 4, atol=1e-10)      # derivative of 2*x² is 4*x
        assert np.isclose(result3[2], 3, atol=1e-10)      # derivative of 1*x³ is 3*x²
    
    def test_fpolyder_edge_cases(self):
        """Test fpolyder edge cases"""
        # Test with single coefficient (constant)
        coeffs = np.array([5])
        result = fpolyder(coeffs)
        assert len(result) == 1  # derivative of constant is [0]
        assert np.isclose(result[0], 0, atol=1e-10)
        
        # Test with two coefficients (linear) - MATLAB: [3, 4] represents 4 + 3*x
        coeffs = np.array([3, 4])
        result = fpolyder(coeffs)
        assert len(result) == 1
        assert np.isclose(result[0], 3, atol=1e-10)  # derivative of 4 + 3*x is 3
        
        # Test with zero coefficients
        coeffs = np.array([0, 0, 0])
        result = fpolyder(coeffs)
        assert len(result) == 2
        assert np.allclose(result, 0, atol=1e-10)
    
    def test_fpolyder_accuracy(self):
        """Test accuracy of polynomial differentiation"""
        # Test with known polynomial and its derivative
        # MATLAB: [2, 3, 4, 5] represents 5 + 4*x + 3*x² + 2*x³
        coeffs = np.array([2, 3, 4, 5])
        
        # Expected derivative: f'(x) = 4 + 6*x + 6*x² = [6, 6, 4] (MATLAB format)
        expected_derivative = np.array([6, 6, 4])
        
        result = fpolyder(coeffs)
        
        # Should match expected derivative
        assert np.allclose(result, expected_derivative, atol=1e-10)
        
        # Test evaluation at specific points
        x_test = np.array([0, 1, 2])
        
        for x in x_test:
            # Evaluate original polynomial
            y_orig = np.polyval(coeffs, x)
            
            # Evaluate derivative
            y_der = np.polyval(result, x)
            
            # Check that derivative is reasonable
            assert np.isfinite(y_der)
    
    def test_fpolyder_different_coefficient_types(self):
        """Test fpolyder with different coefficient types"""
        # Test with integer coefficients
        coeffs_int = np.array([1, 2, 3], dtype=int)
        result_int = fpolyder(coeffs_int)
        assert isinstance(result_int, np.ndarray)
        assert len(result_int) == 2
        
        # Test with float coefficients
        coeffs_float = np.array([1.0, 2.0, 3.0], dtype=float)
        result_float = fpolyder(coeffs_float)
        assert isinstance(result_float, np.ndarray)
        assert len(result_float) == 2
        
        # Test with complex coefficients
        coeffs_complex = np.array([1+1j, 2+2j, 3+3j])
        result_complex = fpolyder(coeffs_complex)
        assert isinstance(result_complex, np.ndarray)
        assert len(result_complex) == 2
        # Note: fpolyder converts to float, so complex parts are discarded
        assert np.all(np.isreal(result_complex))
    
    def test_fpolyder_large_polynomials(self):
        """Test fpolyder with large polynomials"""
        # Test with high order polynomial
        order = 20
        coeffs = np.random.rand(order + 1)
        
        result = fpolyder(coeffs)
        
        # Check dimensions
        assert len(result) == order
        
        # Check that coefficients are real
        assert np.all(np.isreal(result))
        
        # Check that result is finite
        assert np.all(np.isfinite(result))
    
    def test_fpolyder_consistency(self):
        """Test that fpolyder produces consistent results"""
        coeffs = np.array([1, 2, 3, 4, 5])
        
        # Call multiple times
        result1 = fpolyder(coeffs)
        result2 = fpolyder(coeffs)
        
        # Should be consistent
        assert np.allclose(result1, result2, atol=1e-10)
    
    def test_fpolyder_error_handling(self):
        """Test fpolyder error handling"""
        # Test with empty array - should return [0]
        result = fpolyder(np.array([]))
        assert np.array_equal(result, np.array([0]))
        
        # Test with None - should handle gracefully
        result = fpolyder(None)
        assert np.array_equal(result, np.array([0]))
        
        # Test with list input - should work
        result = fpolyder([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
    
    def test_fpolyder_numerical_stability(self):
        """Test numerical stability"""
        # Test with very small coefficients
        coeffs_small = np.array([1e-10, 1e-10, 1e-10])
        result_small = fpolyder(coeffs_small)
        
        # Should still be finite
        assert np.all(np.isfinite(result_small))
        assert len(result_small) == 2
        
        # Test with very large coefficients
        coeffs_large = np.array([1e10, 1e10, 1e10])
        result_large = fpolyder(coeffs_large)
        
        # Should still be finite
        assert np.all(np.isfinite(result_large))
        assert len(result_large) == 2
    
    def test_fpolyder_integration(self):
        """Test integration with polynomial evaluation"""
        # Test that derivative can be used for evaluation
        # MATLAB: [1, 2, 3, 4] represents 4 + 3*x + 2*x² + 1*x³
        coeffs = np.array([1, 2, 3, 4])
        
        # Get derivative
        derivative = fpolyder(coeffs)
        
        # Test evaluation at specific points
        x_test = np.array([0, 1, 2])
        
        for x in x_test:
            # Evaluate derivative
            y_der = np.polyval(derivative, x)
            
            # Should be finite
            assert np.isfinite(y_der)
            
            # For x = 0, derivative should be 3 (coefficient of x term)
            if x == 0:
                assert np.isclose(y_der, 3, atol=1e-10)
    
    def test_fpolyder_mathematical_properties(self):
        """Test mathematical properties of polynomial differentiation"""
        # Test that derivative of sum is sum of derivatives
        coeffs1 = np.array([1, 2, 3])
        coeffs2 = np.array([4, 5, 6])
        
        # Derivative of sum
        sum_coeffs = coeffs1 + coeffs2
        derivative_sum = fpolyder(sum_coeffs)
        
        # Sum of derivatives
        derivative1 = fpolyder(coeffs1)
        derivative2 = fpolyder(coeffs2)
        sum_derivatives = derivative1 + derivative2
        
        # Should be equal
        assert np.allclose(derivative_sum, sum_derivatives, atol=1e-10)
        
        # Test that derivative of constant multiple is constant times derivative
        c = 5.0
        scaled_coeffs = c * coeffs1
        derivative_scaled = fpolyder(scaled_coeffs)
        
        expected_derivative_scaled = c * derivative1
        
        # Should be equal
        assert np.allclose(derivative_scaled, expected_derivative_scaled, atol=1e-10)
