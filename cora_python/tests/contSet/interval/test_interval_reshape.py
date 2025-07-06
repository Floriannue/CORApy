import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalReshape:
    """Test class for interval reshape method"""
    
    def test_reshape_basic(self):
        """Test basic reshape functionality"""
        # Create a 4x1 interval
        I = Interval(np.array([[1], [2], [3], [4]]), np.array([[2], [3], [4], [5]]))
        
        # Reshape to 2x2
        res = I.reshape(2, 2)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Verify shape
        assert res.inf.shape == (2, 2)
        assert res.sup.shape == (2, 2)
        
        # Verify values are preserved (C-order/row-major)
        expected_inf = np.array([[1, 2], [3, 4]])
        expected_sup = np.array([[2, 3], [4, 5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_reshape_with_tuple(self):
        """Test reshape with tuple argument"""
        # Create a 6x1 interval
        I = Interval(np.array([[1], [2], [3], [4], [5], [6]]), 
                    np.array([[2], [3], [4], [5], [6], [7]]))
        
        # Reshape to 2x3 using tuple
        res = I.reshape((2, 3))
        
        # Verify shape
        assert res.inf.shape == (2, 3)
        assert res.sup.shape == (2, 3)
        
        # Verify values (C-order/row-major)
        expected_inf = np.array([[1, 2, 3], [4, 5, 6]])
        expected_sup = np.array([[2, 3, 4], [5, 6, 7]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_reshape_with_list(self):
        """Test reshape with list argument"""
        # Create a 4x1 interval
        I = Interval(np.array([[1], [2], [3], [4]]), np.array([[2], [3], [4], [5]]))
        
        # Reshape to 2x2 using list
        res = I.reshape([2, 2])
        
        # Verify shape
        assert res.inf.shape == (2, 2)
        assert res.sup.shape == (2, 2)
    
    def test_reshape_to_vector(self):
        """Test reshape to vector"""
        # Create a 2x2 interval
        I = Interval(np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]))
        
        # Reshape to 4x1
        res = I.reshape(4, 1)
        
        # Verify shape
        assert res.inf.shape == (4, 1)
        assert res.sup.shape == (4, 1)
        
        # Verify values (C-order/row-major)
        expected_inf = np.array([[1], [2], [3], [4]])
        expected_sup = np.array([[2], [3], [4], [5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_reshape_to_row_vector(self):
        """Test reshape to row vector"""
        # Create a 2x2 interval
        I = Interval(np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]))
        
        # Reshape to 1x4
        res = I.reshape(1, 4)
        
        # Verify shape
        assert res.inf.shape == (1, 4)
        assert res.sup.shape == (1, 4)
        
        # Verify values (C-order/row-major)
        expected_inf = np.array([[1, 2, 3, 4]])
        expected_sup = np.array([[2, 3, 4, 5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_reshape_single_dimension(self):
        """Test reshape with single dimension"""
        # Create a 2x2 interval
        I = Interval(np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]))
        
        # Reshape to 4 elements (1D)
        res = I.reshape(4)
        
        # Verify shape
        assert res.inf.shape == (4,)
        assert res.sup.shape == (4,)
        
        # Verify values (C-order/row-major)
        expected_inf = np.array([1, 2, 3, 4])
        expected_sup = np.array([2, 3, 4, 5])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_reshape_3d(self):
        """Test reshape to 3D"""
        # Create an 8x1 interval
        inf_vals = np.arange(1, 9).reshape(8, 1)
        sup_vals = np.arange(2, 10).reshape(8, 1)
        I = Interval(inf_vals, sup_vals)
        
        # Reshape to 2x2x2
        res = I.reshape(2, 2, 2)
        
        # Verify shape
        assert res.inf.shape == (2, 2, 2)
        assert res.sup.shape == (2, 2, 2)
        
        # Verify values are preserved (using C-order for comparison)
        assert np.allclose(res.inf.flatten(), inf_vals.flatten())
        assert np.allclose(res.sup.flatten(), sup_vals.flatten())
    
    def test_reshape_preserves_properties(self):
        """Test that reshape preserves interval properties"""
        # Create an interval
        I = Interval(np.array([[1], [2], [3], [4]]), np.array([[2], [3], [4], [5]]))
        
        # Reshape
        res = I.reshape(2, 2)
        
        # Volume should be preserved
        assert np.isclose(res.volume_(), I.volume_())
        
        # Number of elements should be the same
        assert res.inf.size == I.inf.size
        assert res.sup.size == I.sup.size
    
    def test_reshape_empty_interval(self):
        """Test reshape with empty interval"""
        I = Interval.empty(4)
        
        # Reshape empty interval to non-empty shape should raise error
        with pytest.raises(ValueError, match="Cannot reshape empty interval"):
            I.reshape(2, 2)
        
        # Reshape empty interval to another empty shape should work
        res = I.reshape(2, 0)  # 2x0 is also empty
        assert res.isemptyobject()
        assert res.inf.shape == (2, 0)
        assert res.sup.shape == (2, 0)
    
    def test_reshape_point_interval(self):
        """Test reshape with point interval"""
        # Create a point interval
        I = Interval(np.array([[1], [2], [3], [4]]), np.array([[1], [2], [3], [4]]))
        
        # Reshape
        res = I.reshape(2, 2)
        
        # Should still be a point interval
        assert np.allclose(res.inf, res.sup)
        assert res.inf.shape == (2, 2)


def test_interval_reshape():
    """Basic test for interval reshape method"""
    # Test with simple interval
    I = Interval(np.array([[1], [2], [3], [4]]), np.array([[2], [3], [4], [5]]))
    
    # Reshape to 2x2
    res = I.reshape(2, 2)
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # Verify shape
    assert res.inf.shape == (2, 2)
    assert res.sup.shape == (2, 2)
    
    # Verify values are preserved (C-order/row-major)
    expected_inf = np.array([[1, 2], [3, 4]])
    expected_sup = np.array([[2, 3], [4, 5]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 