"""
test_emptySet_empty - unit test function of empty instantiation

Syntax:
    res = test_emptySet_empty()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_empty_1d():
    """Test empty instantiation in 1D."""
    # 1D
    n = 1
    O = EmptySet.empty(n)
    assert O.representsa_('emptySet') and O.dim() == 1


def test_empty_5d():
    """Test empty instantiation in 5D."""
    # 5D
    n = 5
    O = EmptySet.empty(n)
    assert O.representsa_('emptySet') and O.dim() == 5


def test_empty_various_dimensions():
    """Test empty instantiation with various dimensions."""
    for n in [2, 3, 4, 7, 10]:
        O = EmptySet.empty(n)
        assert isinstance(O, EmptySet)
        assert O.representsa_('emptySet')
        assert O.dim() == n


def test_empty_static_method():
    """Test that empty is a static method."""
    # Should be callable on class, not just instance
    O = EmptySet.empty(3)
    assert isinstance(O, EmptySet)
    assert O.dim() == 3


if __name__ == "__main__":
    test_empty_1d()
    test_empty_5d()
    test_empty_various_dimensions()
    test_empty_static_method()
    print("All tests passed!") 