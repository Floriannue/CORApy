"""
test_supportFunc - unit test function for supportFunc

Syntax:
    pytest test_supportFunc.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.contSet.supportFunc import supportFunc
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class MockContSet:
    """Mock ContSet for testing supportFunc method"""
    
    def __init__(self, dim_val=2, empty=False):
        self._dim = dim_val
        self._empty = empty
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def representsa_(self, type_str, tol):
        """Mock representsa_ method"""
        if type_str == 'emptySet':
            return self._empty
        return False
    
    def supportFunc_(self, direction, type_, method, max_order_or_splits, tol):
        """Mock implementation returning distance in direction"""
        if self._empty:
            return (-np.inf, np.array([]), np.array([]))
        
        # Simple mock: return dot product with unit direction
        dir_normalized = direction / np.linalg.norm(direction)
        # For 'lower' type, the support function is -h(-l)
        sign = -1 if type_ == 'lower' else 1
        
        # Mock behavior: return normalized direction dot product with unit vector
        # For direction [1,0], normalized = [1,0], dot with [1,1]/sqrt(2) = sqrt(2)/2
        unit_vector = np.ones(self._dim) / np.sqrt(self._dim)
        val = sign * np.dot(dir_normalized.flatten(), unit_vector)
        
        # Return tuple format (val, x, fac) as expected by supportFunc
        return (val, np.ones(self._dim), np.array([1.0]))


class TestSupportFunc:
    """Test class for supportFunc function"""
    
    def test_supportFunc_basic(self):
        """Test basic supportFunc functionality"""
        
        S = MockContSet(2)
        
        # Test with 1D direction vector
        dir = np.array([1, 0])
        result = supportFunc(S, dir)
        expected = np.sqrt(2) / 2  # Normalized [1,0] dot [1,1]
        assert np.isclose(result, expected)
    
    def test_supportFunc_multiple_directions(self):
        """Test supportFunc with multiple individual directions"""
        
        S = MockContSet(2)
        
        # Test with multiple individual directions (not as a matrix)
        dir1 = np.array([1, 0])
        dir2 = np.array([0, 1])
        
        result1 = supportFunc(S, dir1)
        result2 = supportFunc(S, dir2)
        
        # Both should be valid floats
        assert isinstance(result1, (float, np.floating))
        assert isinstance(result2, (float, np.floating))
        
        # Results should be the same for symmetric mock
        assert np.isclose(result1, result2)
    
    def test_supportFunc_format_options(self):
        """Test supportFunc with different format options"""
        
        S = MockContSet(2)
        dir = np.array([1, 1])
        
        # Test with default format ('upper')
        result1 = supportFunc(S, dir)
        
        # Test with explicit format 'lower'
        result2 = supportFunc(S, dir, type_='lower')
        
        assert not np.isclose(result1, result2)
    
    def test_supportFunc_empty_set(self):
        """Test supportFunc with empty set"""
        
        S = MockContSet(2, empty=True)
        dir = np.array([1, 0])
        
        result = supportFunc(S, dir)
        assert result == -np.inf
    
    def test_supportFunc_error_cases(self):
        """Test error cases for supportFunc"""
        
        S = MockContSet(2)
        
        # Test with invalid direction (empty)
        with pytest.raises(CORAerror, match="must be a 2-dimensional column vector"):
            supportFunc(S, np.array([]))
        
        # Test with zero direction
        with pytest.raises(ValueError, match="Direction cannot be the zero vector"):
            supportFunc(S, np.array([0, 0]))
    
    def test_supportFunc_direction_normalization(self):
        """Test that direction normalization works correctly"""
        
        S = MockContSet(2)
        
        # Test with unit vector
        dir1 = np.array([1, 0])
        result1 = supportFunc(S, dir1)
        
        # Test with scaled vector (should give same result)
        dir2 = np.array([2, 0])
        result2 = supportFunc(S, dir2)
        
        assert np.isclose(result1, result2)
    
    def test_supportFunc_high_dimension(self):
        """Test supportFunc with high-dimensional sets"""
        
        S = MockContSet(5)
        dir = np.array([1, 1, 1, 1, 1])
        
        result = supportFunc(S, dir)
        assert isinstance(result, (float, np.floating))
    
    def test_supportFunc_negative_directions(self):
        """Test supportFunc with negative directions"""
        
        S = MockContSet(2)
        
        # Positive direction
        dir_pos = np.array([1, 0])
        result_pos = supportFunc(S, dir_pos)
        
        # Negative direction
        dir_neg = np.array([-1, 0])
        result_neg = supportFunc(S, dir_neg)
        
        # Results should be different
        assert not np.isclose(result_pos, result_neg)


if __name__ == "__main__":
    pytest.main([__file__]) 