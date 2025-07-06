import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalLog:
    """Test class for interval log method"""
    
    def test_log_positive_interval(self):
        """Test log with positive interval"""
        # Create a positive interval
        I = Interval(np.array([[1], [2]]), np.array([[3], [4]]))
        res = I.log()
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Verify bounds (log is increasing)
        expected_inf = np.log(np.array([[1], [2]]))
        expected_sup = np.log(np.array([[3], [4]]))
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_log_small_positive_interval(self):
        """Test log with small positive interval"""
        # Create a small positive interval
        I = Interval(np.array([[0.1], [0.5]]), np.array([[0.5], [1]]))
        res = I.log()
        
        # Verify result
        expected_inf = np.log(np.array([[0.1], [0.5]]))
        expected_sup = np.log(np.array([[0.5], [1]]))
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_log_point_interval(self):
        """Test log with point interval"""
        # Create a point interval
        I = Interval(np.array([[2]]), np.array([[2]]))
        res = I.log()
        
        # For point interval, log should be the log of the point
        expected = np.log(2)
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
    
    def test_log_negative_interval(self):
        """Test log with negative interval (should raise error)"""
        # Create a negative interval
        I = Interval(np.array([[-2], [-1]]), np.array([[-1], [0]]))
        
        with pytest.raises(ValueError) as exc_info:
            I.log()
        assert 'outOfDomain' in str(exc_info.value)
    
    def test_log_mixed_interval(self):
        """Test log with interval crossing zero (should raise error)"""
        # Create an interval that crosses zero
        I = Interval(np.array([[-1], [0.5]]), np.array([[1], [2]]))
        
        with pytest.raises(ValueError) as exc_info:
            I.log()
        assert 'outOfDomain' in str(exc_info.value)
    
    def test_log_zero_boundary(self):
        """Test log with interval touching zero (should raise error)"""
        # Create an interval with zero boundary
        I = Interval(np.array([[0], [1]]), np.array([[1], [2]]))
        
        with pytest.raises(ValueError) as exc_info:
            I.log()
        assert 'outOfDomain' in str(exc_info.value)
    
    def test_log_monotonicity(self):
        """Test that log preserves monotonicity"""
        # log is strictly increasing on (0, ∞)
        I1 = Interval(np.array([[0.1], [0.5]]), np.array([[0.5], [1]]))
        I2 = Interval(np.array([[2], [3]]), np.array([[4], [5]]))
        
        res1 = I1.log()
        res2 = I2.log()
        
        # res1 should be completely less than res2
        assert np.all(res1.sup < res2.inf)
    
    def test_log_empty_interval(self):
        """Test log with empty interval"""
        I = Interval.empty(1)
        res = I.log()
        assert res.isemptyobject()
    
    def test_log_result_properties(self):
        """Test properties of log result"""
        I = Interval(np.array([[1], [2]]), np.array([[3], [4]]))
        res = I.log()
        
        # Result should have same dimension
        assert res.dim() == I.dim()
        
        # Result should contain log of center
        center = I.center()
        log_center = np.log(center)
        contains_result, _, _ = res.contains_(log_center)
        assert contains_result


def test_interval_log():
    """Basic test for interval log method"""
    # Test with simple positive interval
    I = Interval(np.array([[1], [2]]), np.array([[3], [4]]))
    res = I.log()
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # Verify bounds
    expected_inf = np.log(np.array([[1], [2]]))
    expected_sup = np.log(np.array([[3], [4]]))
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 