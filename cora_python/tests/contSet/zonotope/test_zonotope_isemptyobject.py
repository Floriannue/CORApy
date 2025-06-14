# test_zonotope_isemptyobject - unit test function of isemptyobject
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/zonotope/test_zonotope_isemptyobject.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonotopeIsemptyobject:
    """Test class for zonotope isemptyobject method."""

    def test_zonotope_isemptyobject_empty(self):
        """Test isemptyobject with empty zonotope."""
        Z = Zonotope.empty(2)
        assert Z.is_empty()

    def test_zonotope_isemptyobject_zero_center_generators(self):
        """Test isemptyobject with zero center and generators."""
        Z = Zonotope(np.zeros((3, 0)))
        assert Z.is_empty()

    def test_zonotope_isemptyobject_regular(self):
        """Test isemptyobject with regular zonotope."""
        c = np.array([[1], [2]])
        G = np.array([[0.5, 1], [0.5, -1]])
        Z = Zonotope(c, G)
        assert not Z.is_empty()

    def test_zonotope_isemptyobject_point(self):
        """Test isemptyobject with point zonotope (no generators)."""
        c = np.array([[1], [2], [3]])
        Z = Zonotope(c)
        assert not Z.is_empty()

    def test_zonotope_isemptyobject_zero_center(self):
        """Test isemptyobject with zero center but with generators."""
        c = np.zeros((2, 1))
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        assert not Z.is_empty()

    def test_zonotope_isemptyobject_large_dimension(self):
        """Test isemptyobject with large dimension empty zonotope."""
        Z = Zonotope.empty(10)
        assert Z.is_empty()

    def test_zonotope_isemptyobject_large_dimension_non_empty(self):
        """Test isemptyobject with large dimension non-empty zonotope."""
        n = 10
        c = np.ones((n, 1))
        G = np.eye(n)
        Z = Zonotope(c, G)
        assert not Z.is_empty()

    def test_zonotope_isemptyobject_degenerate(self):
        """Test isemptyobject with degenerate zonotope."""
        # Zonotope with linearly dependent generators (still not empty)
        c = np.array([[0], [0]])
        G = np.array([[1, 2], [1, 2]])  # Linearly dependent
        Z = Zonotope(c, G)
        assert not Z.is_empty()

    def test_zonotope_isemptyobject_single_generator(self):
        """Test isemptyobject with single generator."""
        c = np.array([[1], [2]])
        G = np.array([[1], [0]])
        Z = Zonotope(c, G)
        assert not Z.is_empty()


def test_zonotope_isemptyobject():
    """Test function for zonotope isemptyobject method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestZonotopeIsemptyobject()
    test.test_zonotope_isemptyobject_empty()
    test.test_zonotope_isemptyobject_zero_center_generators()
    test.test_zonotope_isemptyobject_regular()
    test.test_zonotope_isemptyobject_point()
    test.test_zonotope_isemptyobject_zero_center()
    test.test_zonotope_isemptyobject_large_dimension()
    test.test_zonotope_isemptyobject_large_dimension_non_empty()
    test.test_zonotope_isemptyobject_degenerate()
    test.test_zonotope_isemptyobject_single_generator()
    
    print("test_zonotope_isemptyobject: all tests passed")


if __name__ == "__main__":
    test_zonotope_isemptyobject() 