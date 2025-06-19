"""
test_combineVec - unit test function for combineVec

Tests the combineVec function for cartesian product combinations.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init import combineVec


class TestCombineVec:
    def test_combineVec_basic(self):
        """Test basic combineVec functionality"""
        Y = combineVec([1, 2], [3, 4])
        expected = np.array([[1, 2, 1, 2], [3, 3, 4, 4]])
        assert np.allclose(Y, expected)
    
    def test_combineVec_three_inputs(self):
        """Test with three inputs"""
        Y = combineVec([1, 2], [3, 4], [5, 6])
        # Should have 2*2*2 = 8 combinations
        assert Y.shape == (3, 8)
        
        # Check first and last columns
        assert np.allclose(Y[:, 0], [1, 3, 5])
        assert np.allclose(Y[:, -1], [2, 4, 6])
    
    def test_combineVec_single_input(self):
        """Test with single input"""
        Y = combineVec([1, 2, 3])
        expected = np.array([[1, 2, 3]])  # Row vector (MATLAB default)
        assert np.allclose(Y, expected)
    
    def test_combineVec_empty_input(self):
        """Test with empty input"""
        Y = combineVec()
        assert Y.size == 0


if __name__ == "__main__":
    pytest.main([__file__]) 