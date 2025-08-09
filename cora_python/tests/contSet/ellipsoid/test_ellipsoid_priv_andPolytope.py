import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.ellipsoid.private.priv_andPolytope import priv_andPolytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_priv_andPolytope_basic_outer_approximation():
    # Basic test for outer approximation
    E = Ellipsoid(np.eye(2), np.array([0, 0]))  # Unit ellipsoid at origin
    # Half-space: x <= 0
    P = Polytope(A=np.array([[1, 0]]), b=np.array([0]))

    # Expected: should be an ellipsoid that is an outer approximation of the intersection
    result_E = priv_andPolytope(E, P, 'outer')

    assert isinstance(result_E, Ellipsoid)
    # The exact numerical assertion can be complex, so check general properties
    assert result_E.dim() == 2
    # Check if the center is shifted towards the half-space
    assert result_E.q[0] <= E.q[0] + E.TOL

def test_priv_andPolytope_outer_multiple_constraints_general_path():
    # General path (multiple inequalities) to hit QP if needed
    E = Ellipsoid(np.eye(2), np.array([0, 0]))
    # Box constraints combine to multiple half-spaces
    P = Polytope(A=np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), b=np.array([1, 1, 1, 1]))
    result_E = priv_andPolytope(E, P, 'outer')
    assert isinstance(result_E, Ellipsoid)
    assert result_E.dim() == 2

def test_priv_andPolytope_basic_inner_approximation():
    # Basic test for inner approximation
    E = Ellipsoid(np.eye(2), np.array([0, 0]))  # Unit ellipsoid at origin
    # Half-space: x >= 0
    P = Polytope(A=np.array([[-1, 0]]), b=np.array([0]))

    # Expected: should be an ellipsoid that is an inner approximation of the intersection
    result_E = priv_andPolytope(E, P, 'inner')

    assert isinstance(result_E, Ellipsoid)
    assert result_E.dim() == 2
    # Check if the center is shifted towards the half-space
    assert result_E.q[0] >= E.q[0] - E.TOL

def test_priv_andPolytope_inner_multiple_constraints_general_path():
    # General path for inner mode
    E = Ellipsoid(np.eye(2), np.array([0, 0]))
    P = Polytope(A=np.array([[1, 1], [-1, 0], [0, -1]]), b=np.array([1.5, 1, 1]))
    result_E = priv_andPolytope(E, P, 'inner')
    assert isinstance(result_E, Ellipsoid)
    assert result_E.dim() == 2

def test_priv_andPolytope_ellipsoid_outside_halfspace():
    # Test case where ellipsoid is completely outside the half-space
    E = Ellipsoid(np.eye(2), np.array([5, 0]))  # Ellipsoid centered at (5,0)
    # Half-space: x <= 0
    P = Polytope(A=np.array([[1, 0]]), b=np.array([0]))

    result_E = priv_andPolytope(E, P, 'outer')
    assert result_E.Q.size == 0 and result_E.q.shape[1] == 0 # Should be an empty ellipsoid

def test_priv_andPolytope_ellipsoid_touching_halfspace():
    # Test case where ellipsoid is touching the half-space
    E = Ellipsoid(np.eye(2), np.array([1, 0]))  # Ellipsoid centered at (1,0), radius 1
    # Half-space: x <= 0
    P = Polytope(A=np.array([[1, 0]]), b=np.array([0]))

    result_E = priv_andPolytope(E, P, 'outer')
    assert isinstance(result_E, Ellipsoid)
    assert np.allclose(result_E.Q, np.zeros((2,2))) # Should be a point ellipsoid
    assert np.allclose(result_E.q, np.array([0,0]).reshape(-1,1))

def test_priv_andPolytope_point_ellipsoid_inside_polytope():
    # Test with a point ellipsoid that is inside the polytope
    E = Ellipsoid(np.zeros((2,2)), np.array([0, 0])) # Point ellipsoid at origin
    P = Polytope(A=np.array([[1, 0], [-1, 0]]), b=np.array([1, 1])) # Box -1 <= x <= 1

    result_E = priv_andPolytope(E, P, 'outer')
    assert isinstance(result_E, Ellipsoid)
    assert np.allclose(result_E.Q, np.zeros((2,2)))
    assert np.allclose(result_E.q, np.array([0,0]).reshape(-1,1))

def test_priv_andPolytope_point_ellipsoid_outside_polytope():
    # Test with a point ellipsoid that is outside the polytope
    E = Ellipsoid(np.zeros((2,2)), np.array([5, 0])) # Point ellipsoid at (5,0)
    P = Polytope(A=np.array([[1, 0], [-1, 0]]), b=np.array([1, 1])) # Box -1 <= x <= 1

    result_E = priv_andPolytope(E, P, 'outer')
    assert result_E.Q.size == 0 and result_E.q.shape[1] == 0

def test_priv_andPolytope_non_full_dim_ellipsoid():
    # Test with a non-full-dimensional ellipsoid (a line segment in 2D)
    E = Ellipsoid(np.array([[1, 0], [0, 0]]), np.array([0, 0])) # Degenerate ellipsoid along x-axis
    P = Polytope(A=np.array([[0, 1]]), b=np.array([0])) # Half-space: y <= 0

    result_E = priv_andPolytope(E, P, 'outer')
    assert isinstance(result_E, Ellipsoid)
    # The result should still be a degenerate ellipsoid
    assert np.linalg.matrix_rank(result_E.Q) <= 1