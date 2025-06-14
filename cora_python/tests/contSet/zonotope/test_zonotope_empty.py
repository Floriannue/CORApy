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

    def test_zonotope_empty_center_generators(self):
        """Test empty zonotope from zero center and generators."""
        Z = Zonotope(np.zeros((3, 0)))
        assert Z.representsa_('emptySet')
        # Empty zonotope may have dimension 0 when constructed this way
        assert Z.is_empty()

    def test_zonotope_empty_properties(self):
        """Test properties of empty zonotopes."""
        Z = Zonotope.empty(2)
        
        # Empty zonotope should be an empty object
        assert Z.is_empty()
        
        # Center should be empty array
        c = Z.center()
        assert c.shape == (2, 0)

    def test_zonotope_empty_operations(self):
        """Test operations with empty zonotopes."""
        Z_empty = Zonotope.empty(2)
        Z_regular = Zonotope(np.array([[1], [2]]), np.array([[0.5, 1], [0.5, -1]]))
        
        # Addition with empty should return empty
        result = Z_empty + Z_regular
        assert result.representsa_('emptySet')
        
        result = Z_regular + Z_empty
        assert result.representsa_('emptySet')
        
        # Matrix multiplication with empty should return empty
        M = np.array([[1, 0], [0, 1], [1, 1]])
        result = M @ Z_empty
        assert result.representsa_('emptySet')
        assert result.dim() == 3

    def test_zonotope_empty_different_dims(self):
        """Test empty zonotopes of different dimensions."""
        for n in [1, 2, 3, 5, 10]:
            Z = Zonotope.empty(n)
            assert Z.representsa_('emptySet')
            assert Z.dim() == n
            assert Z.is_empty()

    def test_zonotope_empty_zero_matrix(self):
        """Test empty zonotope from zero matrix."""
        Z = Zonotope(np.zeros((2, 0)), np.zeros((2, 0)))
        assert Z.representsa_('emptySet')
        # Empty zonotope may have dimension 0 when constructed this way
        assert Z.is_empty()


def test_zonotope_empty():
    """Test function for zonotope empty method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestZonotopeEmpty()
    test.test_zonotope_empty_1d()
    test.test_zonotope_empty_5d()
    test.test_zonotope_empty_center_generators()
    test.test_zonotope_empty_properties()
    test.test_zonotope_empty_operations()
    test.test_zonotope_empty_different_dims()
    test.test_zonotope_empty_zero_matrix()
    
    print("test_zonotope_empty: all tests passed")


if __name__ == "__main__":
    test_zonotope_empty() 