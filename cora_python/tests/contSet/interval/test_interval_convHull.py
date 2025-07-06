import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalConvHull:
    """Test class for interval convHull method"""
    
    def test_convHull_with_interval(self):
        """Test convHull with another interval"""
        # Create two intervals
        I1 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
        
        res = I1.convHull_(I2)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # ConvHull should be the union (for intervals)
        # Expected: union of [-1, 1] and [0, 2] = [-1, 2]
        #           union of [0, 2] and [-1, 1] = [-1, 2]
        expected_inf = np.array([[-1], [-1]])
        expected_sup = np.array([[2], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_convHull_with_point(self):
        """Test convHull with a point (numeric)"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        point = np.array([[0.5], [3]])
        
        res = I.convHull_(point)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # ConvHull should include the point
        # Expected: convHull([-1, 1], 0.5) = [-1, 1] (0.5 is inside)
        #           convHull([0, 2], 3) = [0, 3] (3 extends the interval)
        expected_inf = np.array([[-1], [0]])
        expected_sup = np.array([[1], [3]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_convHull_with_scalar(self):
        """Test convHull with scalar point"""
        # Create a 1D interval
        I = Interval(np.array([[1]]), np.array([[3]]))
        
        res = I.convHull_(0.5)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # ConvHull should extend to include the point
        # Expected: convHull([1, 3], 0.5) = [0.5, 3]
        expected_inf = np.array([[0.5]])
        expected_sup = np.array([[3]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_convHull_point_intervals(self):
        """Test convHull with point intervals"""
        # Create point intervals
        I1 = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[3], [1]]), np.array([[3], [1]]))
        
        res = I1.convHull_(I2)
        
        # ConvHull of two points should be the interval between them
        # Expected: convHull([1, 1], [3, 3]) = [1, 3]
        #           convHull([2, 2], [1, 1]) = [1, 2]
        expected_inf = np.array([[1], [1]])
        expected_sup = np.array([[3], [2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_convHull_empty_interval(self):
        """Test convHull with empty interval"""
        I1 = Interval.empty(2)
        I2 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        
        # ConvHull with empty interval should return the non-empty interval
        res = I1.convHull_(I2)
        assert res.isequal(I2)
        
        # Test the other way
        res = I2.convHull_(I1)
        assert res.isequal(I2)
    
    def test_convHull_both_empty(self):
        """Test convHull with both intervals empty"""
        I1 = Interval.empty(2)
        I2 = Interval.empty(2)
        
        res = I1.convHull_(I2)
        assert res.isemptyobject()
    
    def test_convHull_self(self):
        """Test convHull with itself"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        
        res = I.convHull_(I)
        
        # ConvHull with itself should return the same interval
        assert res.isequal(I)
    
    def test_convHull_overlapping_intervals(self):
        """Test convHull with overlapping intervals"""
        # Create overlapping intervals
        I1 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0], [1]]), np.array([[2], [3]]))
        
        res = I1.convHull_(I2)
        
        # ConvHull should be the union
        # Expected: union of [-1, 1] and [0, 2] = [-1, 2]
        #           union of [0, 2] and [1, 3] = [0, 3]
        expected_inf = np.array([[-1], [0]])
        expected_sup = np.array([[2], [3]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_convHull_commutativity(self):
        """Test that convHull is commutative"""
        I1 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
        
        res1 = I1.convHull_(I2)
        res2 = I2.convHull_(I1)
        
        # ConvHull should be commutative
        assert res1.isequal(res2)


def test_interval_convHull():
    """Basic test for interval convHull method"""
    # Test with simple intervals
    I1 = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    I2 = Interval(np.array([[0], [-1]]), np.array([[2], [1]]))
    
    res = I1.convHull_(I2)
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # ConvHull should be the union
    expected_inf = np.array([[-1], [-1]])
    expected_sup = np.array([[2], [2]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 