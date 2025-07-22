# test_zonotope_empty - unit test function of empty instantiation
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/zonotope/test_zonotope_empty.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonotopeEmpty:
    """Test class for zonotope empty method."""

    def test_zonotope_empty_1d(self):
        """Test empty zonotope in 1D."""
        n = 1
        Z = Zonotope.empty(n)
        assert Z.representsa_('emptySet')
        assert Z.dim() == 1

    def test_zonotope_empty_5d(self):
        """Test empty zonotope in 5D."""
        n = 5
        Z = Zonotope.empty(n)
        assert Z.representsa_('emptySet')
        assert Z.dim() == 5


def test_zonotope_empty():
    """Test function for zonotope empty method.

    Runs all test methods to verify correct implementation.
    """
    test = TestZonotopeEmpty()
    test.test_zonotope_empty_1d()
    test.test_zonotope_empty_5d()

    print("test_zonotope_empty: all tests passed")


if __name__ == "__main__":
    test_zonotope_empty()