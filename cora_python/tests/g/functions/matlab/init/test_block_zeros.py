"""
test_block_zeros - unit test function for block_zeros

Tests the block_zeros function for initializing block zero vectors.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init import block_zeros


class TestBlockZeros:
    def test_block_zeros_single_block(self):
        """Test single block case"""
        blocks = np.array([[1, 5]])
        result = block_zeros(blocks)
        
        # Should return a single vector of size 5
        expected = np.zeros((5, 1))
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, expected)
        assert result.shape == (5, 1)
    
    def test_block_zeros_multiple_blocks(self):
        """Test multiple blocks case"""
        blocks = np.array([[1, 2], [3, 5]])
        result = block_zeros(blocks)
        
        # Should return list of vectors
        assert isinstance(result, list)
        assert len(result) == 2
        
        # First block: size 2-1+1 = 2
        assert result[0].shape == (2, 1)
        assert np.allclose(result[0], np.zeros((2, 1)))
        
        # Second block: size 5-3+1 = 3
        assert result[1].shape == (3, 1)
        assert np.allclose(result[1], np.zeros((3, 1)))
    
    def test_block_zeros_three_blocks(self):
        """Test with three blocks"""
        blocks = np.array([[1, 2], [3, 4], [5, 7]])
        result = block_zeros(blocks)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Block sizes: 2, 2, 3
        assert result[0].shape == (2, 1)
        assert result[1].shape == (2, 1)
        assert result[2].shape == (3, 1)
        
        # All should be zero vectors
        for block in result:
            assert np.allclose(block, np.zeros_like(block))
    
    def test_block_zeros_1d_input(self):
        """Test with 1D input (single block)"""
        blocks = [1, 3]
        result = block_zeros(blocks)
        
        # Should return single vector of size 3
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        assert np.allclose(result, np.zeros((3, 1)))


if __name__ == "__main__":
    pytest.main([__file__]) 