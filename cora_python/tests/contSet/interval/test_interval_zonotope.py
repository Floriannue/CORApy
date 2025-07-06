import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestIntervalZonotope:
    """Test class for interval zonotope method"""
    
    def test_zonotope_basic(self):
        """Test basic zonotope conversion"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        res = I.zonotope()
        
        # Verify result is a Zonotope
        assert isinstance(res, Zonotope)
        
        # Verify the zonotope has correct center
        expected_center = I.center()
        assert np.allclose(res.c, expected_center)
        
        # Verify the zonotope encloses the original interval
        I_back = res.interval()
        assert np.allclose(I_back.inf, I.inf)
        assert np.allclose(I_back.sup, I.sup)
    
    def test_zonotope_multidimensional(self):
        """Test zonotope with multidimensional interval"""
        # Create a 3D interval
        I = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        res = I.zonotope()
        
        # Verify result is a Zonotope
        assert isinstance(res, Zonotope)
        
        # Verify dimensions match
        assert res.dim() == I.dim()
        
        # Verify center
        assert np.allclose(res.c, I.center())
    
    def test_zonotope_point_interval(self):
        """Test zonotope with point interval"""
        # Create a point interval
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        res = I.zonotope()
        
        # Verify result is a Zonotope
        assert isinstance(res, Zonotope)
        
        # For point interval, generators should be zero
        assert np.allclose(res.G, 0)
        
        # Center should be the point
        assert np.allclose(res.c, I.center())
    
    def test_zonotope_empty_interval(self):
        """Test zonotope with empty interval"""
        I = Interval.empty(2)
        res = I.zonotope()
        
        # Verify result is a Zonotope
        assert isinstance(res, Zonotope)
        
        # Should represent empty set
        assert res.isemptyobject()
    
    def test_zonotope_unit_interval(self):
        """Test zonotope with unit interval"""
        # Create a unit interval [-1, 1] x [-1, 1]
        I = Interval(np.array([[-1], [-1]]), np.array([[1], [1]]))
        res = I.zonotope()
        
        # Verify result is a Zonotope
        assert isinstance(res, Zonotope)
        
        # Center should be at origin
        assert np.allclose(res.c, np.array([[0], [0]]))
        
        # Should have diagonal generator matrix
        assert res.G.shape == (2, 2)
        assert np.allclose(np.abs(res.G), np.eye(2))
    
    def test_zonotope_generators(self):
        """Test zonotope generator structure"""
        # Create an interval
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        res = I.zonotope()
        
        # Verify generator matrix has correct dimensions
        n = I.dim()
        assert res.G.shape == (n, n)
        
        # Generators should be diagonal for intervals
        assert np.allclose(res.G, np.diag(np.diag(res.G)))
    
    def test_zonotope_conversion_consistency(self):
        """Test that interval -> zonotope -> interval is consistent"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        
        # Convert to zonotope and back
        Z = I.zonotope()
        I_back = Z.interval()
        
        # Should get back the same interval
        assert np.allclose(I_back.inf, I.inf, atol=1e-14)
        assert np.allclose(I_back.sup, I.sup, atol=1e-14)
    
    def test_zonotope_properties(self):
        """Test properties of converted zonotope"""
        # Create an interval
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        res = I.zonotope()
        
        # Center should be preserved
        assert np.allclose(res.center(), I.center())
        
        # Dimension should be preserved
        assert res.dim() == I.dim()


def test_interval_zonotope():
    """Basic test for interval zonotope method"""
    # Test with simple interval
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    res = I.zonotope()
    
    # Verify result is a Zonotope
    assert isinstance(res, Zonotope)
    
    # Verify center
    expected_center = I.center()
    assert np.allclose(res.c, expected_center)
    
    # Verify dimensions match
    assert res.dim() == I.dim() 