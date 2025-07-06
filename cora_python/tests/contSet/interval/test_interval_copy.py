import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalCopy:
    """Test class for interval copy method"""
    
    def test_copy_basic(self):
        """Test basic copy functionality"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I_copy = I.copy()
        
        # Verify result is an interval
        assert isinstance(I_copy, Interval)
        
        # Verify the copy has the same values
        assert np.allclose(I_copy.inf, I.inf)
        assert np.allclose(I_copy.sup, I.sup)
        
        # Verify they are different objects
        assert I_copy is not I
        assert I_copy.inf is not I.inf
        assert I_copy.sup is not I.sup
    
    def test_copy_independence(self):
        """Test that copy is independent of original"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I_copy = I.copy()
        
        # Modify the original
        I.inf[0] = -5
        I.sup[0] = 5
        
        # Verify the copy is unchanged
        assert I_copy.inf[0] != I.inf[0]
        assert I_copy.sup[0] != I.sup[0]
    
    def test_copy_multidimensional(self):
        """Test copy with multidimensional interval"""
        # Create a 3D interval
        I = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        I_copy = I.copy()
        
        # Verify the copy has the same values
        assert np.allclose(I_copy.inf, I.inf)
        assert np.allclose(I_copy.sup, I.sup)
        assert I_copy.dim() == I.dim()
    
    def test_copy_point_interval(self):
        """Test copy with point interval"""
        # Create a point interval
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        I_copy = I.copy()
        
        # Verify the copy has the same values
        assert np.allclose(I_copy.inf, I.inf)
        assert np.allclose(I_copy.sup, I.sup)
    
    def test_copy_empty_interval(self):
        """Test copy with empty interval"""
        I = Interval.empty(2)
        I_copy = I.copy()
        
        # Verify both are empty
        assert I.isemptyobject()
        assert I_copy.isemptyobject()
        
        # Verify they are different objects
        assert I_copy is not I
    
    def test_copy_preserves_properties(self):
        """Test that copy preserves interval properties"""
        # Create an interval with specific properties
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I_copy = I.copy()
        
        # Verify properties are preserved
        assert np.allclose(I_copy.center(), I.center())
        assert np.allclose(I_copy.rad(), I.rad())
        assert I_copy.dim() == I.dim()
        assert I_copy.volume_() == I.volume_()


def test_interval_copy():
    """Basic test for interval copy method"""
    # Test with simple interval
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    I_copy = I.copy()
    
    # Verify result is an interval
    assert isinstance(I_copy, Interval)
    
    # Verify the copy has the same bounds
    assert np.allclose(I_copy.inf, I.inf)
    assert np.allclose(I_copy.sup, I.sup)
    
    # Verify they are different objects
    assert I_copy is not I 