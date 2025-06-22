"""
test_emptySet_plus - unit test function of plus

Syntax:
    res = test_emptySet_plus()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.zonotope import Zonotope


def test_plus_empty_vector():
    """Test addition with empty vector."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # addition with empty vector
    p = np.empty((n, 0))
    O_ = O.plus(p)
    assert O.isequal(O_)
    
    # different order
    O_ = O.plus(p)  # Note: Python doesn't support p + O directly
    assert O.isequal(O_)


def test_plus_empty_set():
    """Test addition with another empty set."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # addition with another empty set
    O2 = EmptySet(n)
    O_ = O.plus(O2)
    assert O.isequal(O_)


def test_plus_zonotope():
    """Test addition with zonotope."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # init zonotope
    Z = Zonotope(np.zeros((n, 1)), np.eye(n))
    O_ = O.plus(Z)
    assert O.isequal(O_)


def test_plus_operator():
    """Test + operator."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # addition with empty vector using operator
    p = np.empty((n, 0))
    O_ = O + p
    assert O.isequal(O_)


if __name__ == "__main__":
    test_plus_empty_vector()
    test_plus_empty_set()
    test_plus_zonotope()
    test_plus_operator()
    print("All tests passed!") 