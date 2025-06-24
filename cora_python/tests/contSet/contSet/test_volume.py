"""
test_volume - unit test function for volume

Syntax:
    pytest test_volume.py

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
from unittest.mock import Mock
from cora_python.contSet.contSet.volume import volume


class MockContSet:
    """Mock ContSet for testing volume method"""
    
    def __init__(self, dim_val=2, empty=False, vol_value=1.0):
        self._dim = dim_val
        self._empty = empty
        self._vol_value = vol_value
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def volume_(self, method, order):
        """Mock implementation of volume_"""
        if self._empty:
            return 0.0
        
        # Return different volumes based on method
        if method == 'exact':
            return self._vol_value
        elif method == 'reduce':
            return self._vol_value * 0.9  # Approximation
        elif method == 'alamo':
            return self._vol_value * 1.1  # Different approximation
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def representsa_(self, setType, tol):
        return self._empty and setType == 'emptySet'


class TestVolume:
    """Test class for volume function"""
    
    def test_volume_default_parameters(self):
        """Test volume with default parameters"""
        
        S = MockContSet(2, vol_value=4.0)
        result = volume(S)
        
        assert result == 4.0
        assert isinstance(result, float)
    
    def test_volume_exact_method(self):
        """Test volume with exact method"""
        
        S = MockContSet(2, vol_value=2.5)
        result = volume(S, method='exact')
        
        assert result == 2.5
    
    def test_volume_reduce_method(self):
        """Test volume with reduce method"""
        
        S = MockContSet(2, vol_value=2.0)
        result = volume(S, method='reduce')
        
        assert result == 1.8  # 2.0 * 0.9
    
    def test_volume_alamo_method(self):
        """Test volume with alamo method"""
        
        S = MockContSet(2, vol_value=3.0)
        result = volume(S, method='alamo')
        
        assert result == 3.3  # 3.0 * 1.1
    
    def test_volume_invalid_method(self):
        """Test volume with invalid method"""
        
        S = MockContSet(2)
        
        with pytest.raises(ValueError, match="Invalid method"):
            volume(S, method='invalid')
    
    def test_volume_with_order_parameter(self):
        """Test volume with order parameter"""
        
        S = MockContSet(2, vol_value=1.5)
        
        # Valid order values
        result1 = volume(S, order=3)
        assert result1 == 1.5
        
        result2 = volume(S, order=10)
        assert result2 == 1.5
    
    def test_volume_invalid_order(self):
        """Test volume with invalid order parameter"""
        
        S = MockContSet(2)
        
        # Non-positive order
        with pytest.raises(ValueError, match="Order must be a positive integer"):
            volume(S, order=0)
        
        with pytest.raises(ValueError, match="Order must be a positive integer"):
            volume(S, order=-1)
        
        # Non-integer order
        with pytest.raises(ValueError, match="Order must be a positive integer"):
            volume(S, order=2.5)
    
    def test_volume_empty_set(self):
        """Test volume with empty set"""
        
        S = MockContSet(2, empty=True)
        result = volume(S)
        
        assert result == 0.0
    
    def test_volume_exception_handling_empty_set(self):
        """Test volume exception handling when set represents empty set"""
        
        class ExceptionSet(MockContSet):
            def volume_(self, method, order):
                raise RuntimeError("Some error")
            
            def representsa_(self, setType, tol):
                return setType == 'emptySet'
        
        S = ExceptionSet(2, empty=False)
        result = volume(S)
        
        # Should handle exception and return 0.0 for empty set
        assert result == 0.0
    
    def test_volume_exception_handling_non_empty_set(self):
        """Test volume exception handling when set is not empty"""
        
        class ExceptionSet(MockContSet):
            def volume_(self, method, order):
                raise RuntimeError("Some error")
            
            def representsa_(self, setType, tol):
                return False
        
        S = ExceptionSet(2)
        
        # Should re-raise the exception
        with pytest.raises(RuntimeError, match="Some error"):
            volume(S)
    
    def test_volume_zero_result(self):
        """Test volume when result is zero but set is not empty"""
        
        S = MockContSet(2, vol_value=0.0, empty=False)
        result = volume(S)
        
        assert result == 0.0
    
    def test_volume_large_volume(self):
        """Test volume with large volume values"""
        
        S = MockContSet(10, vol_value=1e10)
        result = volume(S)
        
        assert result == 1e10
    
    def test_volume_small_volume(self):
        """Test volume with very small volume values"""
        
        S = MockContSet(2, vol_value=1e-15)
        result = volume(S)
        
        assert result == 1e-15
    
    def test_volume_combined_parameters(self):
        """Test volume with both method and order parameters"""
        
        S = MockContSet(3, vol_value=8.0)
        result = volume(S, method='reduce', order=5)
        
        assert result == 7.2  # 8.0 * 0.9


if __name__ == "__main__":
    pytest.main([__file__]) 