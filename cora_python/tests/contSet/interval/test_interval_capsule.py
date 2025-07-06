import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.capsule.capsule import Capsule
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalCapsule:
    """Test class for interval capsule method"""
    
    def test_capsule_2d_interval(self):
        """Test capsule with 2D interval"""
        # Create a 2D interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        C = I.capsule()
        
        # Verify result is a capsule
        assert isinstance(C, Capsule)
        
        # Verify the capsule encloses the interval
        # The capsule should have the same center as the interval
        assert np.allclose(C.c, I.center())
    
    def test_capsule_1d_interval(self):
        """Test capsule with 1D interval"""
        # Create a 1D interval
        I = Interval(np.array([[-2]]), np.array([[3]]))
        C = I.capsule()
        
        # Verify result is a capsule
        assert isinstance(C, Capsule)
        
        # Verify properties
        assert np.allclose(C.c, I.center())
    
    def test_capsule_3d_interval(self):
        """Test capsule with 3D interval"""
        # Create a 3D interval
        I = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        C = I.capsule()
        
        # Verify result is a capsule
        assert isinstance(C, Capsule)
        
        # Verify the capsule encloses the interval
        assert np.allclose(C.c, I.center())
    
    def test_capsule_point_interval(self):
        """Test capsule with point interval"""
        # Create a point interval (zero width)
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        C = I.capsule()
        
        # Verify result is a capsule
        assert isinstance(C, Capsule)
        
        # For point interval, radius should be zero
        assert np.allclose(C.r, 0)
    
    def test_capsule_empty_interval(self):
        """Test capsule with empty interval"""
        I = Interval.empty(2)
        C = I.capsule()
        
        # Verify result is a capsule
        assert isinstance(C, Capsule)


def test_interval_capsule():
    """Basic test for interval capsule method"""
    # Test with simple 2D interval
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    C = I.capsule()
    
    # Verify result is a capsule
    assert isinstance(C, Capsule)
    
    # Verify the capsule has correct center
    expected_center = np.array([[0], [1]])  # Center of interval
    assert np.allclose(C.c, expected_center) 