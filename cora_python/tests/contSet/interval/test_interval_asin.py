import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalAsin:
    """Test class for interval asin method"""
    
    def test_asin_valid_domain(self):
        """Test asin with valid domain [-1, 1]"""
        # Test case 1: Full valid domain
        I = Interval(np.array([-1]), np.array([1]))
        res = I.asin()
        
        expected_inf = np.arcsin(-1)  # -π/2
        expected_sup = np.arcsin(1)   # π/2
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
        # Test case 2: Partial domain
        I = Interval(np.array([-0.5]), np.array([0.5]))
        res = I.asin()
        
        expected_inf = np.arcsin(-0.5)
        expected_sup = np.arcsin(0.5)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
        # Test case 3: Point interval
        I = Interval(np.array([0.3]), np.array([0.3]))
        res = I.asin()
        
        expected = np.arcsin(0.3)
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
    
    def test_asin_multidimensional(self):
        """Test asin with multidimensional intervals"""
        I = Interval(np.array([[-0.5], [0.2]]), np.array([[0.3], [0.8]]))
        res = I.asin()
        
        expected_inf = np.arcsin(np.array([[-0.5], [0.2]]))
        expected_sup = np.arcsin(np.array([[0.3], [0.8]]))
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_asin_boundary_values(self):
        """Test asin at boundary values"""
        # Test at -1
        I = Interval(np.array([-1]), np.array([-1]))
        res = I.asin()
        expected = np.arcsin(-1)
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
        # Test at 1
        I = Interval(np.array([1]), np.array([1]))
        res = I.asin()
        expected = np.arcsin(1)
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
        # Test at 0
        I = Interval(np.array([0]), np.array([0]))
        res = I.asin()
        expected = np.arcsin(0)
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
    
    def test_asin_invalid_domain(self):
        """Test asin with invalid domain (outside [-1, 1])"""
        # Test case 1: inf < -1 and sup > 1
        I = Interval(np.array([-2]), np.array([2]))
        with pytest.raises(CORAerror) as exc_info:
            I.asin()
        assert 'valid domain' in str(exc_info.value)
        
        # Test case 2: inf > 1
        I = Interval(np.array([1.5]), np.array([2]))
        with pytest.raises(CORAerror) as exc_info:
            I.asin()
        assert 'valid domain' in str(exc_info.value)
        
        # Test case 3: sup < -1
        I = Interval(np.array([-2]), np.array([-1.5]))
        with pytest.raises(CORAerror) as exc_info:
            I.asin()
        assert 'valid domain' in str(exc_info.value)
        
        # Test case 4: inf < -1 and sup in [-1, 1]
        I = Interval(np.array([-1.5]), np.array([0.5]))
        with pytest.raises(CORAerror) as exc_info:
            I.asin()
        assert 'valid domain' in str(exc_info.value)
        
        # Test case 5: inf in [-1, 1] and sup > 1
        I = Interval(np.array([0.5]), np.array([1.5]))
        with pytest.raises(CORAerror) as exc_info:
            I.asin()
        assert 'valid domain' in str(exc_info.value)
    
    def test_asin_mixed_valid_invalid(self):
        """Test asin with mixed valid and invalid intervals"""
        # One valid, one invalid
        I = Interval(np.array([0.5, -2]), np.array([0.8, 2]))
        with pytest.raises(CORAerror) as exc_info:
            I.asin()
        assert 'valid domain' in str(exc_info.value)
    
    def test_asin_monotonicity(self):
        """Test that asin preserves monotonicity"""
        # asin is strictly increasing on [-1, 1]
        I1 = Interval(np.array([-0.8]), np.array([-0.3]))
        I2 = Interval(np.array([0.2]), np.array([0.7]))
        
        res1 = I1.asin()
        res2 = I2.asin()
        
        # res1 should be completely less than res2
        assert np.all(res1.sup < res2.inf)
    
    def test_asin_empty_interval(self):
        """Test asin with empty interval"""
        I = Interval.empty(1)
        res = I.asin()
        assert res.isemptyobject()
    
    def test_asin_result_properties(self):
        """Test properties of asin result"""
        I = Interval(np.array([-0.5]), np.array([0.5]))
        res = I.asin()
        
        # Result should have same dimension
        assert res.dim() == I.dim()
        
        # Result should contain asin of center
        center = I.center()
        asin_center = np.arcsin(center)
        contains_result, _, _ = res.contains_(asin_center)
        assert contains_result


def test_interval_asin():
    """Basic test for interval asin method"""
    # Test with simple interval
    I = Interval(np.array([-0.5]), np.array([0.5]))
    res = I.asin()
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # Verify bounds
    expected_inf = np.arcsin(-0.5)
    expected_sup = np.arcsin(0.5)
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 