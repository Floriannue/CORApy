"""
test_fpolyder - unit test function for fpolyder

Tests the fpolyder function for computing polynomial derivatives.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.polynomial.fpolyder import fpolyder


class TestFpolyder:
    def test_fpolyder_basic(self):
        """Test basic polynomial derivative"""
        # p(x) = x^2 + 2x + 1, derivative = 2x + 2
        p = np.array([1, 2, 1])
        dp = fpolyder(p)
        expected = np.array([2, 2])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_cubic(self):
        """Test cubic polynomial derivative"""
        # p(x) = x^3 + 3x^2 + 3x + 1, derivative = 3x^2 + 6x + 3
        p = np.array([1, 3, 3, 1])
        dp = fpolyder(p)
        expected = np.array([3, 6, 3])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_linear(self):
        """Test linear polynomial derivative"""
        # p(x) = 2x + 5, derivative = 2
        p = np.array([2, 5])
        dp = fpolyder(p)
        expected = np.array([2])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_constant(self):
        """Test constant polynomial derivative"""
        # p(x) = 5, derivative = 0
        p = np.array([5])
        dp = fpolyder(p)
        expected = np.array([0])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_zero_polynomial(self):
        """Test zero polynomial derivative"""
        # p(x) = 0, derivative = 0
        p = np.array([0])
        dp = fpolyder(p)
        expected = np.array([0])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_high_degree(self):
        """Test higher degree polynomial"""
        # p(x) = x^5 + 2x^4 + 3x^3 + 4x^2 + 5x + 6
        # derivative = 5x^4 + 8x^3 + 9x^2 + 8x + 5
        p = np.array([1, 2, 3, 4, 5, 6])
        dp = fpolyder(p)
        expected = np.array([5, 8, 9, 8, 5])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_leading_zeros(self):
        """Test polynomial with leading zeros"""
        # p(x) = 0*x^3 + 0*x^2 + 2x + 1 = 2x + 1, derivative = 2
        p = np.array([0, 0, 2, 1])
        dp = fpolyder(p)
        expected = np.array([2])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_single_term(self):
        """Test single term polynomials"""
        # p(x) = 3x^4, derivative = 12x^3
        p = np.array([3, 0, 0, 0, 0])
        dp = fpolyder(p)
        expected = np.array([12, 0, 0, 0])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_negative_coefficients(self):
        """Test polynomial with negative coefficients"""
        # p(x) = -x^2 + 2x - 1, derivative = -2x + 2
        p = np.array([-1, 2, -1])
        dp = fpolyder(p)
        expected = np.array([-2, 2])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_fractional_coefficients(self):
        """Test polynomial with fractional coefficients"""
        # p(x) = 0.5x^2 + 1.5x + 2.5, derivative = x + 1.5
        p = np.array([0.5, 1.5, 2.5])
        dp = fpolyder(p)
        expected = np.array([1.0, 1.5])
        assert np.allclose(dp, expected)
    
    def test_fpolyder_comparison_with_numpy(self):
        """Test comparison with numpy.polyder for verification"""
        # Test several random polynomials
        for _ in range(10):
            degree = np.random.randint(1, 6)
            p = np.random.randn(degree + 1)
            
            # Our implementation
            dp_ours = fpolyder(p)
            
            # NumPy implementation
            dp_numpy = np.polyder(p)
            
            # Should be identical
            assert np.allclose(dp_ours, dp_numpy)


if __name__ == "__main__":
    pytest.main([__file__]) 