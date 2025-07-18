"""
Test cases for randPoint_ method of EmptySet class
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_randPoint_basic():
    """Test basic randPoint functionality"""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # sample random point
    p = O.randPoint_()
    assert p.shape == (n, 0)


def test_randPoint_different_dimensions():
    """Test randPoint with different dimensions"""
    for n in range(1, 11):  # Test dimensions 1-10
        O = EmptySet(n)
        p = O.randPoint_()
        assert p.shape == (n, 0), f"Failed for dimension {n}"


def test_randPoint_with_parameters():
    """Test randPoint with different parameter combinations"""
    O = EmptySet(3)
    
    # Test with N parameter
    p1 = O.randPoint_(5)
    assert p1.shape == (3, 0)
    
    # Test with type parameter
    p2 = O.randPoint_(5, 'standard')
    assert p2.shape == (3, 0)
    
    # Test with 'all' and 'extreme'
    p3 = O.randPoint_('all', 'extreme')
    assert p3.shape == (3, 0)


def test_randPoint_zero_dimension():
    """Test randPoint with zero dimension"""
    O = EmptySet(0)
    p = O.randPoint_()
    assert p.shape == (0, 0)


def test_randPoint_high_dimension():
    """Test randPoint with high dimension"""
    O = EmptySet(100)
    p = O.randPoint_()
    assert p.shape == (100, 0)


def test_randPoint_return_type():
    """Test that randPoint returns correct type"""
    O = EmptySet(3)
    p = O.randPoint_()
    
    # Should return numpy array
    assert isinstance(p, np.ndarray)
    assert p.shape == (3, 0)


def test_randPoint_empty_array():
    """Test that randPoint returns empty array"""
    O = EmptySet(2)
    p = O.randPoint_()
    
    # Should be empty (0 columns)
    assert p.size == 0
    assert p.shape[1] == 0


def test_randPoint_consistency():
    """Test that randPoint is consistent for same object"""
    O = EmptySet(4)
    
    p1 = O.randPoint_()
    p2 = O.randPoint_()
    
    assert p1.shape == p2.shape == (4, 0) 