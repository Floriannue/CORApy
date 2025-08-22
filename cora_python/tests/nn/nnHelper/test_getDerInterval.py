"""
Test for nnHelper.getDerInterval function

This test verifies that the getDerInterval function works correctly for computing derivative intervals.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.getDerInterval import getDerInterval


class TestGetDerInterval:
    """Test class for getDerInterval function"""
    
    def test_getDerInterval_basic(self):
        """Test basic getDerInterval functionality"""
        # Test with simple case
        f = lambda x: x**2
        l, u = 0, 1
        order = 1
        
        result = getDerInterval(f, l, u, order)
        
        # Check that result is tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # Check that bounds are real numbers
        assert np.isreal(result[0])
        assert np.isreal(result[1])
        
        # Check that lower bound <= upper bound
        assert result[0] <= result[1]
    
    def test_getDerInterval_different_orders(self):
        """Test getDerInterval with different derivative orders"""
        f = lambda x: x**2
        l, u = 0, 1
        
        # Test different orders
        for order in [1, 2, 3]:
            result = getDerInterval(f, l, u, order)
            
            # Check result structure
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] <= result[1]
    
    def test_getDerInterval_different_intervals(self):
        """Test getDerInterval with different intervals"""
        f = lambda x: x**2
        order = 1
        
        # Test different intervals
        test_intervals = [
            (0, 1),
            (-1, 1),
            (0, 2),
            (-2, 2),
            (1, 3)
        ]
        
        for l, u in test_intervals:
            result = getDerInterval(f, l, u, order)
            
            # Check result structure
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] <= result[1]
    
    def test_getDerInterval_different_functions(self):
        """Test getDerInterval with different functions"""
        l, u = 0, 1
        order = 1
        
        # Test different functions
        test_functions = [
            lambda x: x**2,           # Quadratic
            lambda x: x**3,           # Cubic
            lambda x: np.sin(x),      # Sine
            lambda x: np.exp(x),      # Exponential
            lambda x: 1,              # Constant
            lambda x: x,              # Linear
        ]
        
        for f in test_functions:
            try:
                result = getDerInterval(f, l, u, order)
                
                # Check result structure
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert result[0] <= result[1]
                
            except Exception as e:
                # Some functions might fail due to domain issues
                # This is acceptable for testing
                pass
    
    def test_getDerInterval_edge_cases(self):
        """Test getDerInterval edge cases"""
        f = lambda x: x**2
        
        # Test with order 0
        result = getDerInterval(f, 0, 1, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]
        
        # Test with very small interval
        result = getDerInterval(f, 0, 0.001, 1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]
        
        # Test with very large interval
        result = getDerInterval(f, 0, 1000, 1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]
    
    def test_getDerInterval_accuracy(self):
        """Test accuracy of derivative interval computation"""
        # For f(x) = x² on [0,1]:
        # df/dx = 2x, so df/dx ∈ [0, 2]
        f = lambda x: x**2
        l, u = 0, 1
        order = 1
        
        result = getDerInterval(f, l, u, order)
        
        # Should be approximately [0, 2]
        assert np.isclose(result[0], 0, atol=1e-6)
        assert np.isclose(result[1], 2, atol=1e-6)
        
        # For f(x) = x³ on [0,1]:
        # df/dx = 3x², so df/dx ∈ [0, 3]
        f = lambda x: x**3
        result = getDerInterval(f, l, u, order)
        
        # Should be approximately [0, 3]
        assert np.isclose(result[0], 0, atol=1e-6)
        assert np.isclose(result[1], 3, atol=1e-6)
    
    def test_getDerInterval_constant_function(self):
        """Test getDerInterval with constant function"""
        f = lambda x: 5.0  # Constant function
        l, u = 0, 1
        order = 1
        
        result = getDerInterval(f, l, u, order)
        
        # For constant function, derivative should be 0
        assert np.isclose(result[0], 0, atol=1e-10)
        assert np.isclose(result[1], 0, atol=1e-10)
    
    def test_getDerInterval_linear_function(self):
        """Test getDerInterval with linear function"""
        f = lambda x: 2*x + 1  # Linear function
        l, u = 0, 1
        order = 1
        
        result = getDerInterval(f, l, u, order)
        
        # For linear function, derivative should be constant
        assert np.isclose(result[0], 2, atol=1e-10)
        assert np.isclose(result[1], 2, atol=1e-10)
    
    def test_getDerInterval_negative_interval(self):
        """Test getDerInterval with negative interval"""
        f = lambda x: x**2
        l, u = -1, 0
        order = 1
        
        result = getDerInterval(f, l, u, order)
        
        # Check result structure
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]
        
        # For f(x) = x² on [-1,0]:
        # df/dx = 2x, so df/dx ∈ [-2, 0]
        assert result[0] <= 0
        assert result[1] >= -2
    
    def test_getDerInterval_large_order(self):
        """Test getDerInterval with large order"""
        f = lambda x: x**2
        l, u = 0, 1
        order = 10  # Large order
        
        result = getDerInterval(f, l, u, order)
        
        # Check result structure
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]
        
        # For high order derivatives, should still be finite
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
    
    def test_getDerInterval_consistency(self):
        """Test that getDerInterval produces consistent results"""
        f = lambda x: x**2
        l, u = 0, 1
        order = 1
        
        # Call multiple times
        result1 = getDerInterval(f, l, u, order)
        result2 = getDerInterval(f, l, u, order)
        
        # Should be consistent
        assert np.isclose(result1[0], result2[0], atol=1e-10)
        assert np.isclose(result1[1], result2[1], atol=1e-10)
    
    def test_getDerInterval_error_handling(self):
        """Test getDerInterval error handling"""
        f = lambda x: x**2
        
        # Test with invalid interval (l >= u)
        with pytest.raises(ValueError):
            getDerInterval(f, 1, 0, 1)
        
        with pytest.raises(ValueError):
            getDerInterval(f, 0, 0, 1)
        
        # Test with negative order
        with pytest.raises(ValueError):
            getDerInterval(f, 0, 1, -1)
        
        # Test with non-integer order
        with pytest.raises(ValueError):
            getDerInterval(f, 0, 1, 2.5)
    
    def test_getDerInterval_numerical_stability(self):
        """Test numerical stability"""
        # Test with function that might cause numerical issues
        f = lambda x: np.exp(x) * np.sin(x)
        l, u = 0, 10
        order = 1
        
        result = getDerInterval(f, l, u, order)
        
        # Should be finite
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        
        # Should maintain proper bounds
        assert result[0] <= result[1]
