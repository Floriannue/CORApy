import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalMax:
    """Test class for interval max method"""
    
    def test_max_no_arguments(self):
        """Test max with no second argument (returns supremum)"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        res = I.max()
        
        # Should return supremum
        assert np.allclose(res, I.sup)
        assert np.allclose(res, np.array([[1], [2]]))
    
    def test_max_with_scalar(self):
        """Test max with scalar argument"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        res = I.max(0.5)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Expected: max([-1, 1], 0.5) = [0.5, 1], max([0, 2], 0.5) = [0.5, 2]
        expected_inf = np.array([[0.5], [0.5]])
        expected_sup = np.array([[1], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_max_with_negative_scalar(self):
        """Test max with negative scalar"""
        # Create an interval
        I = Interval(np.array([[-2], [-1]]), np.array([[1], [2]]))
        res = I.max(-0.5)
        
        # Expected: max([-2, 1], -0.5) = [-0.5, 1], max([-1, 2], -0.5) = [-0.5, 2]
        expected_inf = np.array([[-0.5], [-0.5]])
        expected_sup = np.array([[1], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_max_with_interval(self):
        """Test max with another interval"""
        # Create two intervals
        I1 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
        
        res = I1.max(I2)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Expected: max([-1, 1], [0, 2]) = [0, 2], max([0, 2], [-1, 1]) = [0, 2]
        expected_inf = np.array([[0], [0]])
        expected_sup = np.array([[2], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_max_point_intervals(self):
        """Test max with point intervals"""
        # Create point intervals
        I1 = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0.5], [3]]), np.array([[0.5], [3]]))
        
        res = I1.max(I2)
        
        # Expected: max([1, 1], [0.5, 0.5]) = [1, 1], max([2, 2], [3, 3]) = [3, 3]
        expected_inf = np.array([[1], [3]])
        expected_sup = np.array([[1], [3]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_max_with_zero(self):
        """Test max with zero"""
        # Create an interval that crosses zero
        I = Interval(np.array([[-2], [-1]]), np.array([[1], [0.5]]))
        res = I.max(0)
        
        # Expected: max([-2, 1], 0) = [0, 1], max([-1, 0.5], 0) = [0, 0.5]
        expected_inf = np.array([[0], [0]])
        expected_sup = np.array([[1], [0.5]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_max_empty_interval(self):
        """Test max with empty interval"""
        I = Interval.empty(1)
        
        # Test with no argument
        res = I.max()
        assert np.allclose(res, I.sup)
        
        # Test with scalar
        res = I.max(5)
        assert isinstance(res, Interval)
    
    def test_max_multidimensional(self):
        """Test max with multidimensional intervals"""
        # Create 3D intervals
        I1 = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        I2 = Interval(np.array([[0], [-1], [1]]), np.array([[2], [1], [2]]))
        
        res = I1.max(I2)
        
        # Verify result
        expected_inf = np.array([[0], [0], [1]])
        expected_sup = np.array([[2], [2], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_max_properties(self):
        """Test mathematical properties of max"""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        
        # max(I, I) should equal I
        res = I.max(I)
        assert res.isequal(I)
        
        # max is commutative for intervals
        I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
        res1 = I.max(I2)
        res2 = I2.max(I)
        assert res1.isequal(res2)


def test_interval_max():
    """Basic test for interval max method"""
    # Test with simple interval and scalar
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    res = I.max(0)
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # Expected: max([-1, 1], 0) = [0, 1], max([0, 2], 0) = [0, 2]
    expected_inf = np.array([[0], [0]])
    expected_sup = np.array([[1], [2]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 