"""
Test constrSat method of zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


def test_constrSat_basic():
    """Test basic constraint satisfaction"""
    # Create a zonotope centered at origin with unit box generators
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))

    # Test constraint x + y <= 2 (should be satisfied)
    C = np.array([[1, 1]])
    d = np.array([2])
    assert Z.constrSat(C, d) == True

    # Test constraint x + y <= 1 (should not be satisfied)
    d = np.array([1])
    assert Z.constrSat(C, d) == False


def test_constrSat_multiple_constraints():
    """Test multiple constraints"""
    # Create a zonotope
    Z = Zonotope(np.array([[1], [1]]), np.array([[0.5, 0], [0, 0.5]]))

    # Test multiple constraints
    C = np.array([[1, 0], [0, 1]])  # x <= d1, y <= d2
    d = np.array([2, 2])  # Both constraints should be satisfied
    assert Z.constrSat(C, d) == True

    d = np.array([1, 1])  # Neither constraint should be satisfied
    assert Z.constrSat(C, d) == False


def test_constrSat_edge_cases():
    """Test edge cases"""
    # Test point zonotope
    Z = Zonotope(np.array([[1], [1]]), np.zeros((2, 0)))
    C = np.array([[1, 1]])
    d = np.array([2])
    assert Z.constrSat(C, d) == True

    # Test tight constraint (boundary case)
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    C = np.array([[1, 1]])
    d = np.array([2])  # The supremum points exactly touch the constraint
    assert Z.constrSat(C, d) == False  # Should be False as we want strict inequality


def test_constrSat_invalid_inputs():
    """Test invalid inputs"""
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))

    # Test incompatible dimensions
    with pytest.raises(Exception):
        C = np.array([[1, 1, 1]])  # Wrong dimension
        d = np.array([2])
        Z.constrSat(C, d)

    # Test invalid d dimension
    with pytest.raises(Exception):
        C = np.array([[1, 1]])
        d = np.array([2, 2])  # Wrong dimension
        Z.constrSat(C, d)