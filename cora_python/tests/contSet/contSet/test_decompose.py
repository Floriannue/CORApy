"""
test_decompose - unit test function for decompose

Syntax:
    pytest test_decompose.py

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
from cora_python.contSet.contSet.decompose import decompose


class MockContSet:
    """Mock ContSet for testing decompose method"""
    
    def __init__(self, dim_val=5, name="MockSet"):
        self._dim = dim_val
        self._name = name
        
    def dim(self):
        return self._dim
    
    def project(self, dims):
        """Mock implementation of project method"""
        # Create a new MockContSet representing the projection
        projected_name = f"Proj{self._name}({dims})"
        projected_set = MockContSet(len(dims), projected_name)
        projected_set._original_dims = dims
        return projected_set
    
    def copy(self):
        """Mock implementation of copy method"""
        return MockContSet(self._dim, f"Copy({self._name})")
    
    def __init__(self, *args):
        """Mock constructor for copy fallback"""
        if len(args) == 1 and isinstance(args[0], MockContSet):
            # Copy constructor
            other = args[0]
            self._dim = other._dim
            self._name = f"Copy({other._name})"
        else:
            # Normal constructor
            self._dim = args[0] if args else 2
            self._name = args[1] if len(args) > 1 else "MockSet"


class TestDecompose:
    """Test class for decompose function"""
    
    def test_decompose_single_block(self):
        """Test decompose with single block (should return copy)"""
        
        S = MockContSet(5, "OriginalSet")
        blocks = np.array([[1, 5]])  # Single block covering all dimensions
        
        result = decompose(S, blocks)
        
        # Should return a copy of the original set
        assert result._dim == 5
        assert "Copy" in result._name
    
    def test_decompose_multiple_blocks(self):
        """Test decompose with multiple blocks"""
        
        S = MockContSet(5, "OriginalSet")
        blocks = np.array([[1, 2], [3, 5]])  # Two blocks: [1,2] and [3,5]
        
        result = decompose(S, blocks)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        # First block should project to dimensions [0, 1] (0-based)
        assert result[0]._dim == 2
        assert hasattr(result[0], '_original_dims')
        assert result[0]._original_dims == [0, 1]
        
        # Second block should project to dimensions [2, 3, 4] (0-based)
        assert result[1]._dim == 3
        assert hasattr(result[1], '_original_dims')
        assert result[1]._original_dims == [2, 3, 4]
    
    def test_decompose_three_blocks(self):
        """Test decompose with three blocks"""
        
        S = MockContSet(6, "OriginalSet")
        blocks = np.array([[1, 2], [3, 4], [5, 6]])  # Three blocks
        
        result = decompose(S, blocks)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Check dimensions of each projected set
        assert result[0]._dim == 2  # Dimensions [0, 1]
        assert result[1]._dim == 2  # Dimensions [2, 3]
        assert result[2]._dim == 2  # Dimensions [4, 5]
    
    def test_decompose_single_dimension_blocks(self):
        """Test decompose with single-dimension blocks"""
        
        S = MockContSet(3, "OriginalSet")
        blocks = np.array([[1, 1], [2, 2], [3, 3]])  # Each dimension as separate block
        
        result = decompose(S, blocks)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Each block should have dimension 1
        for i, projected_set in enumerate(result):
            assert projected_set._dim == 1
            assert projected_set._original_dims == [i]
    
    def test_decompose_overlapping_blocks(self):
        """Test decompose with overlapping blocks"""
        
        S = MockContSet(4, "OriginalSet")
        blocks = np.array([[1, 3], [2, 4]])  # Overlapping blocks
        
        result = decompose(S, blocks)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        # First block: dimensions [0, 1, 2]
        assert result[0]._dim == 3
        assert result[0]._original_dims == [0, 1, 2]
        
        # Second block: dimensions [1, 2, 3]
        assert result[1]._dim == 3
        assert result[1]._original_dims == [1, 2, 3]
    
    def test_decompose_non_consecutive_blocks(self):
        """Test decompose with non-consecutive blocks"""
        
        S = MockContSet(6, "OriginalSet")
        blocks = np.array([[1, 2], [4, 6]])  # Skip dimension 3
        
        result = decompose(S, blocks)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        # First block: dimensions [0, 1]
        assert result[0]._dim == 2
        assert result[0]._original_dims == [0, 1]
        
        # Second block: dimensions [3, 4, 5]
        assert result[1]._dim == 3
        assert result[1]._original_dims == [3, 4, 5]
    
    def test_decompose_list_input(self):
        """Test decompose with list input for blocks"""
        
        S = MockContSet(4, "OriginalSet")
        blocks = [[1, 2], [3, 4]]  # List instead of numpy array
        
        result = decompose(S, blocks)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]._dim == 2
        assert result[1]._dim == 2
    
    def test_decompose_edge_case_full_range(self):
        """Test decompose with block covering full range"""
        
        S = MockContSet(3, "OriginalSet")
        blocks = np.array([[1, 3]])  # Full range as single block
        
        result = decompose(S, blocks)
        
        # Should return copy, not list
        assert not isinstance(result, list)
        assert result._dim == 3
    
    def test_decompose_dimension_indexing(self):
        """Test that dimension indexing is correctly converted from 1-based to 0-based"""
        
        class ProjectionCheckSet(MockContSet):
            def project(self, dims):
                # Store the exact dimensions passed to project
                result = super().project(dims)
                result._exact_dims = dims
                return result
        
        S = ProjectionCheckSet(5, "OriginalSet")
        blocks = np.array([[2, 3], [4, 5]])  # MATLAB-style 1-based indexing
        
        result = decompose(S, blocks)
        
        # Check that dimensions were correctly converted to 0-based
        assert result[0]._exact_dims == [1, 2]  # [2,3] -> [1,2]
        assert result[1]._exact_dims == [3, 4]  # [4,5] -> [3,4]
    
    def test_decompose_large_set(self):
        """Test decompose with large dimensional set"""
        
        S = MockContSet(10, "LargeSet")
        blocks = np.array([[1, 3], [4, 7], [8, 10]])
        
        result = decompose(S, blocks)
        
        assert len(result) == 3
        assert result[0]._dim == 3  # Dimensions [0, 1, 2]
        assert result[1]._dim == 4  # Dimensions [3, 4, 5, 6]
        assert result[2]._dim == 3  # Dimensions [7, 8, 9]
    
    def test_decompose_no_copy_method(self):
        """Test decompose when copy method is not available"""
        
        class NoCopySet(MockContSet):
            def __init__(self, *args):
                if len(args) == 1 and isinstance(args[0], MockContSet):
                    # Copy constructor fallback
                    other = args[0]
                    super().__init__(other._dim, f"ManualCopy({other._name})")
                else:
                    super().__init__(*args)
            
            # Remove copy method
            def copy(self):
                raise AttributeError("'NoCopySet' object has no attribute 'copy'")
        
        S = NoCopySet(3, "NoCopySet")
        blocks = np.array([[1, 3]])  # Single block
        
        # Should handle missing copy method gracefully
        result = decompose(S, blocks)
        assert result._dim == 3


if __name__ == "__main__":
    pytest.main([__file__]) 