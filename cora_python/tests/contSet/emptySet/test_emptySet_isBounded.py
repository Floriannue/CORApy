"""
test_emptySet_isBounded - unit test function of isBounded

Syntax:
    res = test_emptySet_isBounded()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_isBounded_2d():
    """Test boundedness check for 2D empty set."""
    # check boundedness
    O = EmptySet(2)
    assert O.isBounded()


def test_isBounded_various_dimensions():
    """Test boundedness check for various dimensions."""
    # Empty sets are always bounded regardless of dimension
    for n in [0, 1, 3, 5, 10]:
        O = EmptySet(n)
        assert O.isBounded(), f"EmptySet({n}) should be bounded"


def test_isBounded_return_type():
    """Test that isBounded returns boolean."""
    O = EmptySet(4)
    result = O.isBounded()
    assert isinstance(result, bool)
    assert result is True


if __name__ == "__main__":
    test_isBounded_2d()
    test_isBounded_various_dimensions()
    test_isBounded_return_type()
    print("All tests passed!") 