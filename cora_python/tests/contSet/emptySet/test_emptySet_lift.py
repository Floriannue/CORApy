"""
Test cases for lift_ method of EmptySet class
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_lift_basic():
    """Test basic lift functionality"""
    O = EmptySet(4)
    N = 6
    dims = [1, 2, 5, 6]
    
    result = O.lift_(N, dims)
    
    # Should return the same object with modified dimension
    assert result is O
    assert result.dimension == N
    assert result.dim() == N


def test_lift_dimension_change():
    """Test that lift changes the dimension correctly"""
    O = EmptySet(2)
    original_dim = O.dimension
    
    # Lift to higher dimension
    N = 5
    dims = [1, 2]
    result = O.lift_(N, dims)
    
    assert result.dimension == N
    assert result.dimension != original_dim


def test_lift_return_value():
    """Test that lift returns the modified object"""
    O = EmptySet(3)
    N = 7
    dims = [1, 2, 3]
    
    result = O.lift_(N, dims)
    
    # Should return the same object (modified in place)
    assert result is O
    assert result.dimension == N


def test_lift_different_dimensions():
    """Test lift with different dimension combinations"""
    test_cases = [
        (1, 3, [1]),
        (2, 4, [1, 2]),
        (3, 6, [1, 2, 3]),
        (5, 10, [1, 2, 3, 4, 5])
    ]
    
    for original_dim, new_dim, dims in test_cases:
        O = EmptySet(original_dim)
        result = O.lift_(new_dim, dims)
        
        assert result.dimension == new_dim
        assert result.dim() == new_dim 