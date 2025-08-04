"""
Test cases for radius method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet


def test_radius_basic():
    """Test basic radius functionality"""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # compute radius
    r = O.radius()
    assert r == 0


def test_radius_different_dimensions():
    """Test radius with different dimensions"""
    for n in range(1, 11):  # Test dimensions 1-10
        O = EmptySet(n)
        r = O.radius()
        assert r == 0, f"Failed for dimension {n}"


def test_radius_zero_dimension():
    """Test radius with zero dimension"""
    O = EmptySet(0)
    r = O.radius()
    assert r == 0


def test_radius_high_dimension():
    """Test radius with high dimension"""
    O = EmptySet(100)
    r = O.radius()
    assert r == 0


def test_radius_return_type():
    """Test that radius returns correct type"""
    O = EmptySet(3)
    r = O.radius()
    
    # Should return a number (int or float)
    assert isinstance(r, (int, float))
    assert r == 0


def test_radius_consistency():
    """Test that radius is consistent for same object"""
    O = EmptySet(5)
    
    r1 = O.radius()
    r2 = O.radius()
    
    assert r1 == r2 == 0 