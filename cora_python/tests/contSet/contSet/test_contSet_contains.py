"""
test_contains - unit test function for contains

Syntax:
    pytest test_contains.py

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
from unittest.mock import Mock, patch, MagicMock
from cora_python.contSet.contSet.contains import contains


class MockContSet:
    """Mock ContSet for testing contains method"""
    
    def __init__(self, dim_val=2, empty=False):
        self._dim = dim_val
        self._empty = empty
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def contains_(self, S2, method, tol, maxEval, cert_toggle, scaling_toggle):
        # Mock implementation
        if isinstance(S2, np.ndarray):
            # For points, return True for all points within unit square
            if S2.ndim == 1:
                S2 = S2.reshape(-1, 1)
            res = np.all((S2 >= -1) & (S2 <= 1), axis=0)
        else:
            res = True
        cert = True
        scaling = 1.0
        return res, cert, scaling


class TestContains:
    """Test class for contains function"""
    
    def test_contains_basic(self):
        """Test basic contains functionality"""
        
        S1 = MockContSet(2)
        
        # Test with single point inside
        point_inside = np.array([0.5, 0.5])
        result = contains(S1, point_inside)
        assert result == True
        
        # Test with single point outside
        point_outside = np.array([2.0, 2.0])
        result = contains(S1, point_outside)
        assert result == False
    
    def test_contains_multiple_points(self):
        """Test contains with multiple points"""
        
        S1 = MockContSet(2)
        
        # Test with multiple points
        points = np.array([[0.5, 2.0], [0.5, 2.0]])
        result = contains(S1, points)
        expected = np.array([True, False])
        assert np.array_equal(result, expected)
    
    def test_contains_with_cert(self):
        """Test contains with certification"""
        
        S1 = MockContSet(2)
        point = np.array([0.5, 0.5])
        
        result, cert = contains(S1, point, return_cert=True)
        assert result == True
        assert cert == True
    
    def test_contains_with_scaling(self):
        """Test contains with scaling factor"""
        
        S1 = MockContSet(2)
        point = np.array([0.5, 0.5])
        
        result, cert, scaling = contains(S1, point, return_scaling=True)
        assert result == True
        assert cert == True
        assert scaling == 1.0
    
    def test_contains_error_cases(self):
        """Test error cases for contains"""
        
        S1 = MockContSet(2)
        
        # Test invalid method
        point = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="Invalid method"):
            contains(S1, point, method='invalid_method')
        
        # Test invalid tolerance
        with pytest.raises(ValueError, match="Tolerance must be"):
            contains(S1, point, tol=-1)
        
        # Test invalid maxEval
        with pytest.raises(ValueError, match="maxEval must be"):
            contains(S1, point, maxEval=-1)
    
    def test_contains_empty_sets(self):
        """Test contains with empty sets"""
        
        # Empty outer set
        S1 = MockContSet(2, empty=True)
        point = np.array([0.5, 0.5])
        
        result = contains(S1, point)
        assert result == False
        
        # Empty inner set (empty array)
        S1 = MockContSet(2)
        empty_array = np.array([]).reshape(2, 0)
        
        result = contains(S1, empty_array)
        assert result == True
    
    def test_contains_default_parameters(self):
        """Test contains with default parameters"""
        
        S1 = MockContSet(2)
        point = np.array([0.5, 0.5])
        
        # Test with default method, tol, maxEval
        result = contains(S1, point)
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__]) 