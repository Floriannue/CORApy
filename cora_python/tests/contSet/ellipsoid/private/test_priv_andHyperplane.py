import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.ellipsoid.private.priv_andHyperplane import priv_andHyperplane
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestPrivAndHyperplane:

    def test_empty_intersection(self):
        # Ellipsoid at origin, radius 1
        E = Ellipsoid(np.eye(2), np.zeros((2, 1)))
        # Hyperplane x=5, no intersection
        P = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1, 0]]), np.array([[5]]))
        
        result_E = priv_andHyperplane(E, P)
        assert result_E.isemptyobject()

    def test_full_intersection_non_degenerate_2d(self):
        # Ellipsoid at origin, Q=I
        E = Ellipsoid(np.eye(2), np.zeros((2, 1)), 1e-6)
        # Hyperplane x + y = 0 (passes through origin)
        P = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1, 1]]), np.array([[0]]))

        result_E = priv_andHyperplane(E, P)
        # Expected: A degenerate ellipsoid (a line segment) with new Q and q
        # For x+y=0, x=-y. Q = [[1,0],[0,1]], q=[[0],[0]]
        # Intersection will be x^2+y^2 <= 1 and x=-y => 2x^2 <= 1 => x^2 <= 0.5
        # This should be an ellipsoid with radius sqrt(0.5) in the direction [1,-1]
        # The resulting ellipsoid will be 1D within the 2D space.
        
        assert not result_E.isemptyobject()
        assert result_E.dim() == 2 # Still in 2D space
        assert result_E.rank() == 1 # Degenerate to 1D

        # Check if center is origin
        assert np.allclose(result_E.q, np.zeros((2,1)), atol=1e-9)

        # Check if the primary axis matches [1/sqrt(2), -1/sqrt(2)] direction
        # The intersection line segment goes from (-sqrt(0.5), sqrt(0.5)) to (sqrt(0.5), -sqrt(0.5))
        # For the degenerate ellipsoid equation (x-q)^T Q^+ (x-q) <= 1 to be satisfied by boundary points,
        # the eigenvalue should be 1.0 (verified by mathematical analysis)
        evals, evecs = np.linalg.eigh(result_E.Q)
        # Find the non-zero eigenvalue and its eigenvector
        non_zero_eval_idx = np.argmax(np.abs(evals))
        assert np.isclose(evals[non_zero_eval_idx], 1.0, atol=1e-9) # Correct eigenvalue for degenerate ellipsoid
        # The eigenvector should be parallel to [1, -1] or [-1, 1]
        expected_dir = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]])
        actual_dir = evecs[:, non_zero_eval_idx].reshape(-1,1)
        # Check if parallel (dot product of normalized vectors is +/- 1)
        assert np.isclose(np.abs(np.dot(expected_dir.flatten(), actual_dir.flatten())), 1.0, atol=1e-9)


    def test_intersection_0d_ellipsoid_on_hyperplane(self):
        # 0-dimensional ellipsoid (a point)
        E = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
        # Hyperplane x + y = 3 (contains the point (1,2))
        P = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1, 1]]), np.array([[3]]))

        result_E = priv_andHyperplane(E, P)
        assert not result_E.isemptyobject()
        assert result_E.rank() == 0
        assert np.array_equal(result_E.q, E.q)

    def test_intersection_0d_ellipsoid_off_hyperplane(self):
        # 0-dimensional ellipsoid (a point)
        E = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
        # Hyperplane x + y = 5 (does not contain the point (1,2))
        P = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1, 1]]), np.array([[5]]))

        result_E = priv_andHyperplane(E, P)
        assert result_E.isemptyobject()

    def test_1d_case(self):
        # 1D ellipsoid: [0.5, 0.5], Q = [[0.25]] (radius 0.5)
        E = Ellipsoid(np.array([[0.25]]), np.array([[0.5]]))
        # 1D hyperplane: x = 0.5
        P = Polytope(np.array([[]]).reshape(0,1), np.array([[]]).reshape(0,1), np.array([[1]]), np.array([[0.5]]))

        result_E = priv_andHyperplane(E, P)
        # Should be a 0-dimensional ellipsoid (a point) at x=0.5
        assert not result_E.isemptyobject()
        assert result_E.rank() == 0
        assert np.isclose(result_E.q[0,0], 0.5, atol=1e-9)
        assert np.allclose(result_E.Q, np.zeros((1,1)), atol=1e-9)

    def test_degenerate_ellipsoid_intersection(self):
        # Degenerate ellipsoid: a line segment along x-axis from (-1,0) to (1,0)
        # Q = [[1, 0], [0, 0]], q = [[0],[0]]
        E = Ellipsoid(np.array([[1, 0], [0, 0]]), np.array([[0], [0]]), 1e-6)
        # Hyperplane: y = 0.5 (parallel to x-axis, shifted up)
        P = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[0, 1]]), np.array([[0.5]]))

        result_E = priv_andHyperplane(E, P)
        # This should be an empty set because the ellipsoid is on y=0 and hyperplane is y=0.5
        assert result_E.isemptyobject()

    def test_degenerate_ellipsoid_intersection_on_plane(self):
        # Degenerate ellipsoid: a line segment along x-axis from (-1,0) to (1,0)
        # Q = [[1, 0], [0, 0]], q = [[0],[0]]
        E = Ellipsoid(np.array([[1, 0], [0, 0]]), np.array([[0], [0]]), 1e-6)
        # Hyperplane: y = 0 (same plane as ellipsoid)
        P = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[0, 1]]), np.array([[0]]))

        result_E = priv_andHyperplane(E, P)
        # Should be the original ellipsoid itself, as it's fully contained in the plane
        assert not result_E.isemptyobject()
        assert np.allclose(result_E.Q, E.Q, atol=1e-9)
        assert np.allclose(result_E.q, E.q, atol=1e-9)

    # Test the 'a < -E.TOL' error case (unlikely to hit with simple inputs without numerical issues, but good to have)
    def test_error_case_a_negative(self):
        # This case is hard to construct deterministically without inducing numerical instability.
        # It happens when the intermediate 'a' value (from [1] Kurzhanski) becomes significantly negative.
        # For now, we will skip or provide a placeholder. If encountered in real scenarios,
        # we'd debug the specific numerical inputs that trigger it.
        pass 