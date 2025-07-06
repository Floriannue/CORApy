import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalInfimum:
    """Test class for interval infimum method"""
    
    def test_infimum_basic(self):
        """Test basic infimum functionality"""
        # Create an interval
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        inf_result = I.infimum()
        
        # Verify result equals the inf property
        assert np.allclose(inf_result, I.inf)
        assert np.allclose(inf_result, np.array([[-1], [0]]))
    
    def test_infimum_multidimensional(self):
        """Test infimum with multidimensional interval"""
        # Create a 3D interval
        I = Interval(np.array([[-1], [0], [-0.5]]), np.array([[1], [2], [1.5]]))
        inf_result = I.infimum()
        
        # Verify result
        expected = np.array([[-1], [0], [-0.5]])
        assert np.allclose(inf_result, expected)
        assert np.allclose(inf_result, I.inf)
    
    def test_infimum_point_interval(self):
        """Test infimum with point interval"""
        # Create a point interval
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        inf_result = I.infimum()
        
        # For point interval, infimum equals the point
        expected = np.array([[1], [2]])
        assert np.allclose(inf_result, expected)
        assert np.allclose(inf_result, I.inf)
        assert np.allclose(inf_result, I.sup)
    
    def test_infimum_negative_values(self):
        """Test infimum with negative values"""
        # Create interval with negative bounds
        I = Interval(np.array([[-5], [-3]]), np.array([[-2], [-1]]))
        inf_result = I.infimum()
        
        # Verify result
        expected = np.array([[-5], [-3]])
        assert np.allclose(inf_result, expected)
        assert np.allclose(inf_result, I.inf)
    
    def test_infimum_mixed_signs(self):
        """Test infimum with mixed positive/negative values"""
        # Create interval crossing zero
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        inf_result = I.infimum()
        
        # Verify result
        expected = np.array([[-2], [1]])
        assert np.allclose(inf_result, expected)
        assert np.allclose(inf_result, I.inf)
    
    def test_infimum_return_type(self):
        """Test that infimum returns numpy array"""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        inf_result = I.infimum()
        
        # Verify return type
        assert isinstance(inf_result, np.ndarray)
        assert inf_result.shape == I.inf.shape
    
    def test_infimum_empty_interval(self):
        """Test infimum with empty interval"""
        I = Interval.empty(2)
        inf_result = I.infimum()
        
        # Should return the inf property of empty interval
        assert np.allclose(inf_result, I.inf)

    def test_infimum_independence(self):
        """Test that infimum result is independent of original"""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        inf_result = I.infimum()
        
        # Modify the result
        original_inf = I.inf.copy()
        inf_result[0] = -999
        
        # Verify original interval is unchanged
        assert np.allclose(I.inf, original_inf)


def test_interval_infimum():
    """Basic test for interval infimum method"""
    # Test with simple interval
    I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
    inf_result = I.infimum()
    
    # Verify result
    expected = np.array([[-1], [0]])
    assert np.allclose(inf_result, expected)
    assert np.allclose(inf_result, I.inf) 