"""
test_reorder_numeric - unit test function for reorder_numeric

Syntax:
    pytest test_reorder_numeric.py

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
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric


class TestReorderNumeric:
    """Test class for reorder_numeric function"""
    
    def test_reorder_numeric_basic(self):
        """Test basic reorder_numeric functionality"""
        
        # Test with simple array
        arr = np.array([3, 1, 4, 1, 5])
        result = reorder_numeric(arr)
        expected = np.array([1, 1, 3, 4, 5])
        assert np.array_equal(result, expected)
        
        # Test with negative numbers
        arr = np.array([-2, 3, -1, 0])
        result = reorder_numeric(arr)
        expected = np.array([-2, -1, 0, 3])
        assert np.array_equal(result, expected)
        
        # Test with floats
        arr = np.array([3.14, 1.41, 2.71])
        result = reorder_numeric(arr)
        expected = np.array([1.41, 2.71, 3.14])
        assert np.allclose(result, expected)
    
    def test_reorder_numeric_2d(self):
        """Test reorder_numeric with 2D arrays"""
        
        # Test with 2D array (should sort along specified axis)
        arr = np.array([[3, 1, 4], [2, 5, 1]])
        
        # Default behavior - might sort flattened or along axis
        result = reorder_numeric(arr)
        
        # Check that result has same shape
        assert result.shape == arr.shape
        
        # Check that all elements are preserved
        assert np.array_equal(np.sort(arr.flatten()), np.sort(result.flatten()))
    
    def test_reorder_numeric_edge_cases(self):
        """Test edge cases for reorder_numeric"""
        
        # Empty array
        arr = np.array([])
        result = reorder_numeric(arr)
        assert result.size == 0
        
        # Single element
        arr = np.array([42])
        result = reorder_numeric(arr)
        expected = np.array([42])
        assert np.array_equal(result, expected)
        
        # Already sorted
        arr = np.array([1, 2, 3, 4, 5])
        result = reorder_numeric(arr)
        expected = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(result, expected)
        
        # Reverse sorted
        arr = np.array([5, 4, 3, 2, 1])
        result = reorder_numeric(arr)
        expected = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(result, expected)
    
    def test_reorder_numeric_duplicates(self):
        """Test reorder_numeric with duplicate values"""
        
        # Array with duplicates
        arr = np.array([3, 1, 3, 2, 1])
        result = reorder_numeric(arr)
        expected = np.array([1, 1, 2, 3, 3])
        assert np.array_equal(result, expected)
        
        # All same values
        arr = np.array([5, 5, 5, 5])
        result = reorder_numeric(arr)
        expected = np.array([5, 5, 5, 5])
        assert np.array_equal(result, expected)
    
    def test_reorder_numeric_special_values(self):
        """Test reorder_numeric with special float values"""
        
        # Test with inf and -inf
        arr = np.array([1, np.inf, -np.inf, 0])
        result = reorder_numeric(arr)
        
        # Check that -inf comes first, inf comes last
        assert result[0] == -np.inf
        assert result[-1] == np.inf
        assert np.isfinite(result[1])
        assert np.isfinite(result[2])
        
        # Test with NaN (behavior might vary)
        arr = np.array([1, np.nan, 2, 3])
        result = reorder_numeric(arr)
        
        # NaN handling might vary, just check shape is preserved
        assert result.shape == arr.shape
    
    def test_reorder_numeric_large_array(self):
        """Test reorder_numeric with larger arrays"""
        
        # Generate random array
        np.random.seed(42)
        arr = np.random.randn(1000)
        result = reorder_numeric(arr)
        
        # Check that result is sorted
        assert np.allclose(result, np.sort(arr))
        
        # Check that all elements are preserved
        assert np.allclose(np.sort(arr), np.sort(result))
    
    def test_reorder_numeric_different_dtypes(self):
        """Test reorder_numeric with different data types"""
        
        # Integer array
        arr = np.array([3, 1, 4, 1, 5], dtype=int)
        result = reorder_numeric(arr)
        expected = np.array([1, 1, 3, 4, 5])
        assert np.array_equal(result, expected)
        
        # Float array
        arr = np.array([3.0, 1.0, 4.0], dtype=float)
        result = reorder_numeric(arr)
        expected = np.array([1.0, 3.0, 4.0])
        assert np.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__]) 