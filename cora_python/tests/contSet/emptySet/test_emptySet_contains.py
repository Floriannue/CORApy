"""
test_emptySet_contains - unit test function of contains_

Syntax:
    res = test_emptySet_contains()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.zonotope import Zonotope


def test_contains_empty_points():
    """Test containment of empty point set."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # empty point set
    p = np.empty((n, 0))
    assert O.contains(p)


def test_contains_single_point():
    """Test containment of single point."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # single point
    p = np.array([[2], [1]])
    assert not O.contains(p)


def test_contains_multiple_points():
    """Test containment of multiple points."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # multiple points
    p = np.array([[1, 2, 3], [4, 5, 6]])
    assert not O.contains(p)


def test_contains_zonotope():
    """Test containment of zonotope."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # zonotope
    Z = Zonotope(np.zeros((n, 1)), np.eye(n))
    assert not O.contains(Z)


def test_contains_empty_set():
    """Test containment of another empty set."""
    # init empty sets
    n = 2
    O1 = EmptySet(n)
    O2 = EmptySet(n)
    
    # empty set contains empty set
    assert O1.contains(O2)


def test_contains_various_dimensions():
    """Test containment for various dimensions."""
    for n in [1, 3, 5]:
        O = EmptySet(n)
        
        # Empty point set should be contained
        p_empty = np.empty((n, 0))
        assert O.contains(p_empty)
        
        # Non-empty point should not be contained
        p_nonempty = np.ones((n, 1))
        assert not O.contains(p_nonempty)
        
        # Another empty set should be contained
        O2 = EmptySet(n)
        assert O.contains(O2)


if __name__ == "__main__":
    test_contains_empty_points()
    test_contains_single_point()
    test_contains_multiple_points()
    test_contains_zonotope()
    test_contains_empty_set()
    test_contains_various_dimensions()
    print("All tests passed!") 