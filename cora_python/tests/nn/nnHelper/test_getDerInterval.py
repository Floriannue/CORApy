"""
Test for nnHelper.getDerInterval function

This test verifies that the getDerInterval function works correctly for computing derivative intervals.
Based on MATLAB test: test_nn_nnHelper_gerDerInterval.m
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.getDerInterval import getDerInterval


class TestGetDerInterval:
    """Test class for getDerInterval function"""
    
    def test_getDerInterval_basic(self):
        """Test basic getDerInterval functionality - matches MATLAB test"""
        l = -1
        u = 3
        
        # Linear case: 2x + 3
        coeffs = np.array([2, 3])
        derl, deru = getDerInterval(coeffs, l, u)
        
        # Derivative of 2x + 3 is 2 (constant)
        assert np.isclose(derl, 2)
        assert np.isclose(deru, 2)
        
        # Linear case: -5x + 1
        coeffs = np.array([-5, 1])
        derl, deru = getDerInterval(coeffs, l, u)
        
        # Derivative of -5x + 1 is -5 (constant)
        assert np.isclose(derl, -5)
        assert np.isclose(deru, -5)
    
    def test_getDerInterval_different_orders(self):
        """Test getDerInterval with different polynomial orders"""
        l = -1
        u = 3
        
        # Quadratic case: x^2 + 2
        coeffs = np.array([1, 0, 2])
        derl, deru = getDerInterval(coeffs, l, u)
        
        # Derivative of x^2 + 2 is 2x, so derivative ranges from -2 to 6
        assert np.isclose(derl, -2, atol=1e-10)
        assert np.isclose(deru, 6, atol=1e-10)
        
        # Quadratic case: x^2 + 6x + 2
        coeffs = np.array([1, 6, 2])
        derl, deru = getDerInterval(coeffs, l, u)
        
        # Derivative of x^2 + 6x + 2 is 2x + 6, so derivative ranges from 4 to 12
        assert np.isclose(derl, 4, atol=1e-10)
        assert np.isclose(deru, 12, atol=1e-10)
        
        # Quadratic case: -x^2 + 7x + 2
        coeffs = np.array([-1, 7, 2])
        derl, deru = getDerInterval(coeffs, l, u)
        
        # Derivative of -x^2 + 7x + 2 is -2x + 7, so derivative ranges from 1 to 9
        assert np.isclose(derl, 1, atol=1e-10)
        assert np.isclose(deru, 9, atol=1e-10)
    
    def test_getDerInterval_different_intervals(self):
        """Test getDerInterval with different intervals"""
        # Test with symmetric interval
        coeffs = np.array([1, 0, 0])  # x^2
        derl, deru = getDerInterval(coeffs, -2, 2)
        
        # Derivative of x^2 is 2x, so derivative ranges from -4 to 4
        assert np.isclose(derl, -4, atol=1e-10)
        assert np.isclose(deru, 4, atol=1e-10)
        
        # Test with positive interval
        derl, deru = getDerInterval(coeffs, 0, 3)
        
        # Derivative ranges from 0 to 6
        assert np.isclose(derl, 0, atol=1e-10)
        assert np.isclose(deru, 6, atol=1e-10)
        
        # Test with negative interval
        derl, deru = getDerInterval(coeffs, -3, 0)
        
        # Derivative ranges from -6 to 0
        assert np.isclose(derl, -6, atol=1e-10)
        assert np.isclose(deru, 0, atol=1e-10)
    
    def test_getDerInterval_edge_cases(self):
        """Test getDerInterval edge cases"""
        # Constant polynomial
        coeffs = np.array([5])
        derl, deru = getDerInterval(coeffs, 0, 1)
        
        # Derivative of constant is 0
        assert np.isclose(derl, 0)
        assert np.isclose(deru, 0)
        
        # Single point interval
        coeffs = np.array([1, 2, 3])  # x^2 + 2x + 3
        derl, deru = getDerInterval(coeffs, 1, 1)
        
        # At x=1, derivative is 2x + 2 = 4
        assert np.isclose(derl, 4)
        assert np.isclose(deru, 4)
    
    def test_getDerInterval_accuracy(self):
        """Test getDerInterval accuracy with cubic polynomial"""
        l = -1
        u = 3
        
        # Cubic case: x^3 + 0.1x^2 + 2
        coeffs = np.array([1, 0.1, 0, 2])
        derl, deru = getDerInterval(coeffs, l, u)
        
        # Derivative is 3x^2 + 0.2x
        # At x=-1: 3 - 0.2 = 2.8
        # At x=3: 27 + 0.6 = 27.6
        # Minimum occurs at x = -0.2/6 = -1/30 ≈ -0.0333
        # At x=-1/30: 3*(1/900) + 0.2*(-1/30) = 1/300 - 2/300 = -1/300
        assert np.isclose(derl, -1/300, atol=1e-10)
        assert np.isclose(deru, 27.6, atol=1e-10)
    
    def test_getDerInterval_constant_function(self):
        """Test getDerInterval with constant function"""
        coeffs = np.array([7])
        derl, deru = getDerInterval(coeffs, -5, 5)
        
        # Derivative of constant is 0
        assert np.isclose(derl, 0)
        assert np.isclose(deru, 0)
    
    def test_getDerInterval_linear_function(self):
        """Test getDerInterval with linear function"""
        coeffs = np.array([3, 4])  # 3x + 4
        derl, deru = getDerInterval(coeffs, -2, 2)
        
        # Derivative of 3x + 4 is 3 (constant)
        assert np.isclose(derl, 3)
        assert np.isclose(deru, 3)
    
    def test_getDerInterval_negative_interval(self):
        """Test getDerInterval with negative interval"""
        coeffs = np.array([1, 0, 0])  # x^2
        derl, deru = getDerInterval(coeffs, -5, -1)
        
        # Derivative of x^2 is 2x, so derivative ranges from -10 to -2
        assert np.isclose(derl, -10, atol=1e-10)
        assert np.isclose(deru, -2, atol=1e-10)
    
    def test_getDerInterval_large_order(self):
        """Test getDerInterval with large polynomial order"""
        # 4th degree polynomial: x^4 - 2x^2 + 1
        coeffs = np.array([1, 0, -2, 0, 1])
        derl, deru = getDerInterval(coeffs, -2, 2)
        
        # Derivative is 4x^3 - 4x = 4x(x^2 - 1)
        # Critical points at x = -1, 0, 1
        # At x=-2: 4(-8) - 4(-2) = -32 + 8 = -24
        # At x=-1: 4(-1) - 4(-1) = -4 + 4 = 0
        # At x=0: 0
        # At x=1: 4(1) - 4(1) = 4 - 4 = 0
        # At x=2: 4(8) - 4(2) = 32 - 8 = 24
        assert np.isclose(derl, -24, atol=1e-10)
        assert np.isclose(deru, 24, atol=1e-10)
    
    def test_getDerInterval_consistency(self):
        """Test getDerInterval consistency"""
        coeffs = np.array([1, 2, 3])  # x^2 + 2x + 3
        l, u = -1, 1
        
        # Multiple calls should give same results
        derl1, deru1 = getDerInterval(coeffs, l, u)
        derl2, deru2 = getDerInterval(coeffs, l, u)
        
        assert np.isclose(derl1, derl2)
        assert np.isclose(deru1, deru2)
    
    def test_getDerInterval_error_handling(self):
        """Test getDerInterval error handling"""
        coeffs = np.array([1, 2, 3])
        
        # Invalid interval (l > u)
        with pytest.raises(ValueError):
            getDerInterval(coeffs, 1, 0)
    
    def test_getDerInterval_numerical_stability(self):
        """Test getDerInterval numerical stability"""
        # Test with very small coefficients
        coeffs = np.array([1e-10, 1e-12, 1e-15])
        derl, deru = getDerInterval(coeffs, -1, 1)
        
        # Should still return finite values
        assert np.isfinite(derl)
        assert np.isfinite(deru)
        assert derl <= deru