"""
Test cases for vertices_ method of EmptySet class
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_vertices_basic():
    """Test basic vertices functionality"""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # check vertices size
    V = O.vertices_()
    assert V.shape == (2, 0)


def test_vertices_different_dimensions():
    """Test vertices with different dimensions"""
    for n in range(1, 11):  # Test dimensions 1-10
        O = EmptySet(n)
        V = O.vertices_()
        assert V.shape == (n, 0), f"Failed for dimension {n}"


def test_vertices_zero_dimension():
    """Test vertices with zero dimension"""
    O = EmptySet(0)
    V = O.vertices_()
    assert V.shape == (0, 0)


def test_vertices_high_dimension():
    """Test vertices with high dimension"""
    O = EmptySet(100)
    V = O.vertices_()
    assert V.shape == (100, 0)


def test_vertices_return_type():
    """Test that vertices returns correct type"""
    O = EmptySet(3)
    V = O.vertices_()
    
    # Should return numpy array
    assert isinstance(V, np.ndarray)
    assert V.shape == (3, 0)


def test_vertices_empty_array():
    """Test that vertices returns empty array"""
    O = EmptySet(2)
    V = O.vertices_()
    
    # Should be empty (0 columns)
    assert V.size == 0
    assert V.shape[1] == 0


def test_vertices_zeros_array():
    """Test that vertices returns zeros array"""
    O = EmptySet(2)
    V = O.vertices_()
    
    # Should be zeros array (even though empty)
    assert np.array_equal(V, np.zeros((2, 0)))


def test_vertices_consistency():
    """Test that vertices is consistent for same object"""
    O = EmptySet(4)
    
    V1 = O.vertices_()
    V2 = O.vertices_()
    
    assert V1.shape == V2.shape == (4, 0)
    assert np.array_equal(V1, V2) 