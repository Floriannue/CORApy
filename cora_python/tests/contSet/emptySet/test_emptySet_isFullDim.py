"""
Test cases for isFullDim method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet


def test_isFullDim_basic():
    """Test basic isFullDim functionality"""
    O = EmptySet(2)
    result = O.isFullDim()
    
    # Should always return False for empty sets
    assert result is False


def test_isFullDim_different_dimensions():
    """Test isFullDim with different dimensions"""
    for dim in [1, 2, 3, 5, 10]:
        O = EmptySet(dim)
        result = O.isFullDim()
        
        # Should always return False regardless of dimension
        assert result is False


def test_isFullDim_return_type():
    """Test that isFullDim returns correct type"""
    O = EmptySet(3)
    result = O.isFullDim()
    
    assert isinstance(result, bool) 