import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalSupremum:
    """Test class for interval supremum method"""
    
    def test_supremum_basic(self):
        """Test basic supremum functionality"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        sup_result = I.supremum()
        
        # Verify result equals the sup property
        assert np.allclose(sup_result, I.sup)
        assert np.allclose(sup_result, np.array([[1], [2]]))
    
    def test_supremum_multidimensional(self):
        """Test supremum with multidimensional interval"""
        # Create a 3D interval
        I = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        sup_result = I.supremum()
        
        # Verify result
        expected = np.array([[1], [2], [1.5]])
        assert np.allclose(sup_result, expected)
        assert np.allclose(sup_result, I.sup)
    
    def test_supremum_point_interval(self):
        """Test supremum with point interval"""
        # Create a point interval
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        sup_result = I.supremum()
        
        # For point interval, supremum equals the point
        expected = np.array([[1], [2]])
        assert np.allclose(sup_result, expected)
        assert np.allclose(sup_result, I.sup)
        assert np.allclose(sup_result, I.inf)
    
    def test_supremum_negative_values(self):
        """Test supremum with negative values"""
        # Create interval with negative bounds
        I = Interval(np.array([[-5], [-3]]), np.array([[-2], [-1]]))
        sup_result = I.supremum()
        
        # Verify result
        expected = np.array([[-2], [-1]])
        assert np.allclose(sup_result, expected)
        assert np.allclose(sup_result, I.sup)
    
    def test_supremum_mixed_signs(self):
        """Test supremum with mixed positive/negative values"""
        # Create interval crossing zero
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        sup_result = I.supremum()
        
        # Verify result
        expected = np.array([[3], [4]])
        assert np.allclose(sup_result, expected)
        assert np.allclose(sup_result, I.sup)
    
    def test_supremum_return_type(self):
        """Test that supremum returns numpy array"""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        sup_result = I.supremum()
        
        # Verify return type
        assert isinstance(sup_result, np.ndarray)
        assert sup_result.shape == I.sup.shape
    
    def test_supremum_empty_interval(self):
        """Test supremum with empty interval"""
        I = Interval.empty(2)
        sup_result = I.supremum()
        
        # Should return the sup property of empty interval
        assert np.allclose(sup_result, I.sup)

    def test_supremum_independence(self):
        """Test that supremum result is independent of original"""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        sup_result = I.supremum()
        
        # Modify the result
        original_sup = I.sup.copy()
        sup_result[0] = 999
        
        # Verify original interval is unchanged
        assert np.allclose(I.sup, original_sup)


def test_interval_supremum():
    """Basic test for interval supremum method"""
    # Test with simple interval
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    sup_result = I.supremum()
    
    # Verify result
    expected = np.array([[1], [2]])
    assert np.allclose(sup_result, expected)
    assert np.allclose(sup_result, I.sup) 