"""
test_ellipsoid_isemptyobject - unit tests for ellipsoid/isemptyobject

Syntax:
    python -m pytest cora_python/tests/contSet/ellipsoid/test_ellipsoid_isemptyobject.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.isemptyobject import isemptyobject


class TestEllipsoidIsemptyobject:

    def test_isemptyobject_empty_ellipsoid(self):
        """Test isemptyobject for a truly empty ellipsoid."""
        # An empty ellipsoid is defined by empty Q and q, and default TOL (1e-6)
        Q_empty = np.array([[]]).reshape(0, 0)
        q_empty = np.zeros((1, 0)) # q should be an n x 0 zero vector (e.g. 1x0 for 1D, 2x0 for 2D etc.)
        E_empty = Ellipsoid(Q_empty, q_empty, 1e-6)
        assert isemptyobject(E_empty), "Expected an empty ellipsoid to be recognized as empty."

    def test_isemptyobject_empty_ellipsoid_tol_empty_array(self):
        """Test isemptyobject for an empty ellipsoid with TOL as empty array."""
        Q_empty = np.array([[]]).reshape(0, 0)
        q_empty = np.array([[]]).reshape(0, 1)
        E_empty = Ellipsoid(Q_empty, q_empty, np.array([])) # TOL as empty array
        assert isemptyobject(E_empty), "Expected an empty ellipsoid with empty TOL to be recognized as empty."

    def test_isemptyobject_non_empty_q(self):
        """Test isemptyobject when q is not empty."""
        Q_empty = np.array([[]]).reshape(0, 0)
        q_non_empty = np.array([[1], [2]])
        E_non_empty = Ellipsoid(Q_empty, q_non_empty, 1e-6)
        assert not isemptyobject(E_non_empty), "Expected ellipsoid with non-empty q to be non-empty."

    def test_isemptyobject_non_empty_Q(self):
        """Test isemptyobject when Q is not empty."""
        Q_non_empty = np.eye(2)
        q_empty = np.array([[]]).reshape(0, 1)
        E_non_empty = Ellipsoid(Q_non_empty, q_empty, 1e-6)
        assert not isemptyobject(E_non_empty), "Expected ellipsoid with non-empty Q to be non-empty."

    def test_isemptyobject_non_default_tol(self):
        """Test isemptyobject when TOL is not the default value (1e-6)."""
        Q_empty = np.array([[]]).reshape(0, 0)
        q_empty = np.array([[]]).reshape(0, 1)
        E_non_empty = Ellipsoid(Q_empty, q_empty, 1e-7) # Non-default TOL
        assert not isemptyobject(E_non_empty), "Expected ellipsoid with non-default TOL to be non-empty."

    def test_isemptyobject_regular_ellipsoid(self):
        """Test isemptyobject for a regular, non-empty ellipsoid."""
        Q = np.array([[2, 1], [1, 2]])
        q = np.array([[0], [0]])
        E = Ellipsoid(Q, q, 1e-6)
        assert not isemptyobject(E), "Expected a regular ellipsoid to be non-empty."


if __name__ == '__main__':
    pytest.main([__file__]) 