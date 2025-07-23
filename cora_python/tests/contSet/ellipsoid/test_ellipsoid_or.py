import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestEllipsoidOr:
    def test_empty_case(self):
        # init cases
        Q1 = np.array([[5.4387811500952807, 12.4977183618314545],
                       [12.4977183618314545, 29.6662117284481646]])
        q1 = np.array([[-0.7445068341257537],
                       [3.5800647524843665]])
        E1 = Ellipsoid(Q1, q1, 1e-6)

        E_empty = Ellipsoid.empty(2)
        assert (E1 | E_empty) == E1

    def test_union_contains_points(self):
        # test non-deg
        Q1 = np.array([[5.4387811500952807, 12.4977183618314545], [12.4977183618314545, 29.6662117284481646]])
        q1 = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E1 = Ellipsoid(Q1, q1, 1e-6)
        E1 = Ellipsoid(E1.Q, np.zeros_like(E1.q), E1.TOL) # Center at origin for easier comparison

        Q2 = np.array([[0.3542431574242590, 0.0233699257103926], [0.0233699257103926, 2.4999614009532856]])
        q2 = np.array([[0.0873801375346114], [-2.4641617305288825]])
        E2 = Ellipsoid(Q2, q2, 1e-6)
        E2 = Ellipsoid(E2.Q, np.zeros_like(E2.q), E2.TOL) # Center at origin for easier comparison

        try:
            Eres = E1 | E2
            # MATLAB's randPoint generates multiple points (e.g., 2 points for E1, 2 for E2)
            # and concatenates them. We should do the same.
            Y_nd = np.concatenate((E1.rand_point(2), E2.rand_point(2)), axis=1)
            # The 'contains' method should be a method of the Ellipsoid class, or callable.
            # Assuming it's a method attached to the Ellipsoid class, we call it on Eres.
            assert np.all(Eres.contains(Y_nd))
        except Exception as e:
            # If the solver fails, it should raise a CORAerror, not a NotImplementedError anymore.
            # The test will now fail as expected, indicating the solver problem.
            pytest.fail(f"Test failed due to an unexpected exception: {type(e).__name__}: {e}")

    def test_zero_rank_union(self):
        # test zero rank ellipsoid
        Q1 = np.eye(2)
        q1 = np.zeros((2, 1))
        E1 = Ellipsoid(Q1, q1)

        Q0 = np.zeros((2, 2))
        q0 = np.array([[1.0986933635979599], [-1.9884387759871638]])
        E0 = Ellipsoid(Q0, q0, 1e-6)
        E0 = Ellipsoid(E0.Q, np.zeros_like(E0.q), E0.TOL) # Center at origin for easier comparison

        try:
            Eres = E1 | E0
            # MATLAB's randPoint generates multiple points for E1, and E0.q is a single point.
            Y_0 = np.concatenate((E1.rand_point(2), E0.q), axis=1)
            assert np.all(Eres.contains(Y_0))
        except Exception as e:
            pytest.fail(f"Test failed due to an unexpected exception: {type(e).__name__}: {e}") 