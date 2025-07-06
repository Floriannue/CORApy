import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


class TestIntervalZonoBundle:
    """Test class for interval zonoBundle method"""
    
    def test_zonoBundle_basic(self):
        """Test basic zonoBundle conversion"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        res = I.zonoBundle()
        
        # Verify result is a ZonoBundle
        assert isinstance(res, ZonoBundle)
        
        # Verify the ZonoBundle encloses the original interval
        # Convert back to interval to check
        I_back = res.interval()
        
        # Check containment using the public contains method (returns boolean)
        assert I_back.contains(I.inf)
        assert I_back.contains(I.sup)
    
    def test_zonoBundle_multidimensional(self):
        """Test zonoBundle with multidimensional interval"""
        # Create a 3D interval
        I = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        res = I.zonoBundle()
        
        # Verify result is a ZonoBundle
        assert isinstance(res, ZonoBundle)
        
        # Verify dimensions match
        assert res.dim() == I.dim()
    
    def test_zonoBundle_point_interval(self):
        """Test zonoBundle with point interval"""
        # Create a point interval
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        res = I.zonoBundle()
        
        # Verify result is a ZonoBundle
        assert isinstance(res, ZonoBundle)
        
        # Should represent a point
        assert res.dim() == I.dim()
    
    def test_zonoBundle_empty_interval(self):
        """Test zonoBundle with empty interval"""
        I = Interval.empty(2)
        res = I.zonoBundle()
        
        # Verify result is a ZonoBundle
        assert isinstance(res, ZonoBundle)
        
        # Should represent empty set
        assert res.isemptyobject()
    
    def test_zonoBundle_center_preserved(self):
        """Test that zonoBundle preserves center"""
        # Create an interval
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        res = I.zonoBundle()
        
        # Center should be preserved
        assert np.allclose(res.center(), I.center())


def test_interval_zonoBundle():
    """Basic test for interval zonoBundle method"""
    # Test with simple interval
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    res = I.zonoBundle()
    
    # Verify result is a ZonoBundle
    assert isinstance(res, ZonoBundle)
    
    # Verify dimensions match
    assert res.dim() == I.dim() 