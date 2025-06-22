"""
test_emptySet_and - unit test function of and_

Syntax:
    res = test_emptySet_and()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.zonotope import Zonotope


def test_and_self():
    """Test intersection with itself."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # intersection with itself
    O_ = O.and_(O)
    assert O.isequal(O_)


def test_and_zonotope():
    """Test intersection with zonotope."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # init zonotope
    Z = Zonotope(np.zeros((n, 1)), np.eye(n))
    
    # intersection with zonotope
    O_ = O.and_(Z)
    assert O.isequal(O_)


def test_and_operator():
    """Test & operator."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # intersection with itself using operator
    O_ = O & O
    assert O.isequal(O_)


if __name__ == "__main__":
    test_and_self()
    test_and_zonotope()
    test_and_operator()
    print("All tests passed!") 