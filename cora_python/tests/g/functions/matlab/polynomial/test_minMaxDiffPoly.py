"""
test_minMaxDiffPoly - unit test function for minMaxDiffPoly

Tests the minMaxDiffPoly function for computing min/max polynomial differences.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.polynomial import minMaxDiffPoly


class TestMinMaxDiffPoly:
    def test_minMaxDiffPoly_basic(self):
        """Test basic minMaxDiffPoly functionality"""
        # p1(x) = x^2, p2(x) = x, difference = x^2 - x = x(x-1)
        # On [0, 2]: minimum at x=0.5 (value -0.25), maximum at x=2 (value 2)
        coeffs1 = [1, 0, 0]  # x^2
        coeffs2 = [1, 0]     # x
        l, u = 0, 2
        
        diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
        
        # Expected: min at x=0.5 gives -0.25, max at x=2 gives 2
        assert np.isclose(diffl, -0.25, atol=1e-10)
        assert np.isclose(diffu, 2.0, atol=1e-10)
    
    def test_minMaxDiffPoly_constant_difference(self):
        """Test with constant difference"""
        # p1(x) = x + 2, p2(x) = x, difference = 2 (constant)
        coeffs1 = [1, 2]  # x + 2
        coeffs2 = [1, 0]  # x
        l, u = -1, 1
        
        diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
        
        # Constant difference of 2
        assert np.isclose(diffl, 2.0, atol=1e-10)
        assert np.isclose(diffu, 2.0, atol=1e-10)
    
    def test_minMaxDiffPoly_same_polynomials(self):
        """Test with identical polynomials"""
        # p1(x) = p2(x) = x^2 + x + 1, difference = 0
        coeffs1 = [1, 1, 1]
        coeffs2 = [1, 1, 1]
        l, u = -2, 2
        
        diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
        
        # Difference should be zero everywhere
        assert np.isclose(diffl, 0.0, atol=1e-10)
        assert np.isclose(diffu, 0.0, atol=1e-10)
    
    def test_minMaxDiffPoly_cubic(self):
        """Test with cubic polynomial"""
        # p1(x) = x^3, p2(x) = 0, difference = x^3
        # On [-1, 1]: minimum at x=-1 (value -1), maximum at x=1 (value 1)
        coeffs1 = [1, 0, 0, 0]  # x^3
        coeffs2 = [0]           # 0
        l, u = -1, 1
        
        diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
        
        assert np.isclose(diffl, -1.0, atol=1e-10)
        assert np.isclose(diffu, 1.0, atol=1e-10)
    
    def test_minMaxDiffPoly_different_lengths(self):
        """Test with polynomials of different degrees"""
        # p1(x) = x^3 + x, p2(x) = x^2, difference = x^3 - x^2 + x = x(x^2 - x + 1) = x(x-0.5)^2 + 0.75x
        coeffs1 = [1, 0, 1, 0]  # x^3 + x
        coeffs2 = [1, 0, 0]     # x^2
        l, u = 0, 1
        
        diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
        
        # Should find extrema correctly
        assert diffl <= diffu
        assert isinstance(diffl, (int, float))
        assert isinstance(diffu, (int, float))


if __name__ == "__main__":
    pytest.main([__file__]) 