import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalMin:
    """Test class for interval min method"""
    
    def test_min_no_arguments(self):
        """Test min with no second argument (returns infimum)"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        res = I.min()
        
        # Should return infimum
        assert np.allclose(res, I.inf)
        assert np.allclose(res, np.array([[-1], [0]]))
    
    def test_min_with_scalar(self):
        """Test min with scalar argument"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        res = I.min(0.5)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Expected: min([-1, 1], 0.5) = [-1, 0.5], min([0, 2], 0.5) = [0, 0.5]
        expected_inf = np.array([[-1], [0]])
        expected_sup = np.array([[0.5], [0.5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_min_with_negative_scalar(self):
        """Test min with negative scalar"""
        # Create an interval
        I = Interval(np.array([[-2], [-1]]), np.array([[1], [2]]))
        res = I.min(-0.5)
        
        # Expected: min([-2, 1], -0.5) = [-2, -0.5], min([-1, 2], -0.5) = [-1, -0.5]
        expected_inf = np.array([[-2], [-1]])
        expected_sup = np.array([[-0.5], [-0.5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_min_with_interval(self):
        """Test min with another interval"""
        # Create two intervals
        I1 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
        
        res = I1.min(I2)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Expected: min([-1, 1], [0, 2]) = [-1, 1], min([0, 2], [-1, 1]) = [-1, 1]
        expected_inf = np.array([[-1], [-1]])
        expected_sup = np.array([[1], [1]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_min_point_intervals(self):
        """Test min with point intervals"""
        # Create point intervals
        I1 = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0.5], [3]]), np.array([[0.5], [3]]))
        
        res = I1.min(I2)
        
        # Expected: min([1, 1], [0.5, 0.5]) = [0.5, 0.5], min([2, 2], [3, 3]) = [2, 2]
        expected_inf = np.array([[0.5], [2]])
        expected_sup = np.array([[0.5], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_min_with_zero(self):
        """Test min with zero"""
        # Create an interval that crosses zero
        I = Interval(np.array([[-2], [-1]]), np.array([[1], [0.5]]))
        res = I.min(0)
        
        # Expected: min([-2, 1], 0) = [-2, 0], min([-1, 0.5], 0) = [-1, 0]
        expected_inf = np.array([[-2], [-1]])
        expected_sup = np.array([[0], [0]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_min_empty_interval(self):
        """Test min with empty interval"""
        I = Interval.empty(1)
        
        # Test with no argument
        res = I.min()
        assert np.allclose(res, I.inf)
        
        # Test with scalar
        res = I.min(5)
        assert isinstance(res, Interval)
    
    def test_min_multidimensional(self):
        """Test min with multidimensional intervals"""
        # Create 3D intervals
        I1 = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        I2 = Interval(np.array([[0], [-1], [1]]), np.array([[2], [1], [2]]))
        
        res = I1.min(I2)
        
        # Verify result
        expected_inf = np.array([[-1], [-1], [-0.5]])
        expected_sup = np.array([[1], [1], [1.5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_min_properties(self):
        """Test mathematical properties of min"""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        
        # min(I, I) should equal I
        res = I.min(I)
        assert res.isequal(I)
        
        # min is commutative for intervals
        I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
        res1 = I.min(I2)
        res2 = I2.min(I)
        assert res1.isequal(res2)


def test_interval_min():
    """Basic test for interval min method"""
    # Test with simple interval and scalar
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    res = I.min(0)
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # Expected: min([-1, 1], 0) = [-1, 0], min([0, 2], 0) = [0, 0]
    expected_inf = np.array([[-1], [0]])
    expected_sup = np.array([[0], [0]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 