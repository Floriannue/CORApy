"""
test_ellipsoid_isequal - unit test function of isequal

This module tests the ellipsoid isequal implementation.

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.zonotope.zonotope import Zonotope # For testing against other contSet types
from cora_python.contSet.emptySet import EmptySet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestEllipsoidIsequal:

    def create_ellipsoid(self, Q, q=None, TOL=1e-6):
        if q is None:
            q = np.zeros((Q.shape[0], 1))
        return Ellipsoid(Q, q, TOL)

    # MATLAB example test case
    def test_isequal_matlab_example(self):
        E1 = self.create_ellipsoid(np.array([[1,0],[0,0.5]]), np.array([[1],[1]]))
        E2 = self.create_ellipsoid(np.array([[1+1e-15,0],[0,0.5]]), np.array([[1],[1]]))
        res = E1.isequal(E2)
        assert res == True

    # Exact match
    def test_isequal_exact_match(self):
        E1 = self.create_ellipsoid(np.eye(2), np.array([[1],[2]]))
        E2 = self.create_ellipsoid(np.eye(2), np.array([[1],[2]]))
        res = E1.isequal(E2)
        assert res == True

    # Different but within tolerance (Q)
    def test_isequal_within_q_tolerance(self):
        E1 = self.create_ellipsoid(np.eye(2), np.array([[1],[2]]))
        E2 = self.create_ellipsoid(np.eye(2) * (1 + 1e-8), np.array([[1],[2]]))
        res = E1.isequal(E2, tol=1e-7)
        assert res == True

    # Different but within tolerance (q)
    def test_isequal_within_q_tolerance_center(self):
        E1 = self.create_ellipsoid(np.eye(2), np.array([[1],[2]]))
        E2 = self.create_ellipsoid(np.eye(2), np.array([[1+1e-8],[2]]))
        res = E1.isequal(E2, tol=1e-7)
        assert res == True

    # Different, outside tolerance (Q)
    def test_isequal_outside_q_tolerance(self):
        E1 = self.create_ellipsoid(np.eye(2))
        E2 = self.create_ellipsoid(np.eye(2) * 1.1)
        res = E1.isequal(E2)
        assert res == False

    # Different, outside tolerance (q)
    def test_isequal_outside_q_tolerance_center(self):
        E1 = self.create_ellipsoid(np.eye(2), np.array([[1],[2]]))
        E2 = self.create_ellipsoid(np.eye(2), np.array([[10],[2]]))
        res = E1.isequal(E2)
        assert res == False

    # Dimension mismatch
    def test_isequal_dimension_mismatch(self):
        E1 = self.create_ellipsoid(np.eye(2))
        E2 = self.create_ellipsoid(np.eye(3))
        res = E1.isequal(E2)
        assert res == False

    # One empty, one not
    def test_isequal_one_empty_one_not(self):
        E1 = self.create_ellipsoid(np.array([[]]).reshape(0,0), np.array([[]]).reshape(0,1))
        E2 = self.create_ellipsoid(np.eye(2))
        res = E1.isequal(E2)
        assert res == False

    # Both empty
    def test_isequal_both_empty(self):
        E1 = self.create_ellipsoid(np.array([[]]).reshape(0,0), np.array([[]]).reshape(0,1))
        E2 = self.create_ellipsoid(np.array([[]]).reshape(0,0), np.array([[]]).reshape(0,1))
        res = E1.isequal(E2)
        assert res == True

    # Ellipsoid vs numeric (scalar, matching dimension, representing a point)
    def test_isequal_ellipsoid_vs_scalar_point_1d(self):
        E = self.create_ellipsoid(np.zeros((1,1)), np.array([[5]]))
        scalar_val = 5.0
        res = E.isequal(scalar_val)
        assert res == True

    # Ellipsoid vs numeric (vector, matching dimension, representing a point)
    def test_isequal_ellipsoid_vs_vector_point_2d(self):
        E = self.create_ellipsoid(np.zeros((2,2)), np.array([[1],[2]]))
        vec_val = np.array([[1],[2]])
        res = E.isequal(vec_val)
        assert res == True

    # Ellipsoid vs numeric (scalar, matching dim, but not a point -> False)
    def test_isequal_ellipsoid_vs_scalar_not_point(self):
        E = self.create_ellipsoid(np.eye(1), np.array([[0]]))
        scalar_val = 5.0
        res = E.isequal(scalar_val)
        assert res == False # Not a point representation

    # Ellipsoid vs numeric (vector, mismatching dimension -> CORAerror wrongValue)
    def test_isequal_ellipsoid_vs_numeric_mismatch_throws_error(self):
        E = self.create_ellipsoid(np.eye(2))
        numeric_val = np.array([[1,2,3]]) # Mismatch in dim
        with pytest.raises(CORAerror) as excinfo:
            E.isequal(numeric_val)
        assert excinfo.value.identifier == 'CORA:wrongValue'

    # Ellipsoid vs unsupported contSet (e.g., Zonotope) -> CORAerror noops
    def test_isequal_ellipsoid_vs_unsupported_contset_throws_error(self):
        E = self.create_ellipsoid(np.eye(2))
        Z = Zonotope(np.array([[0],[0]]), np.array([[1,0],[0,1]]))
        with pytest.raises(CORAerror) as excinfo:
            E.isequal(Z)
        assert excinfo.value.identifier == 'CORA:noops'

    # Test precedence (e.g., if S has lower precedence than E, S.isequal(E) should be called)
    # This requires mocking/understanding precedence if not explicitly defined in Zonotope/Interval
    # For now, we'll test direct call, assuming precedence dispatch works.
    # A deeper test would involve creating mock ContSet subclasses with different precedences.
    # Given `Ellipsoid.precedence = 50` and default `ContSet.precedence = 50`, 
    # other `contSet` types might have different precedence values.
    # This part of the test might require more advanced setup.
    # For simplicity, we'll focus on the direct `E.isequal(S)` call, expecting `CORA:noops` for unknown types. 