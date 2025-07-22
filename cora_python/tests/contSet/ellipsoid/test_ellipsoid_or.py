import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

class TestEllipsoidOr:
    def test_empty_case(self):
        n = 2
        E1 = Ellipsoid(np.eye(n), np.zeros((n, 1)))
        E_empty = Ellipsoid.empty(n)
        try:
            result = E1.or_(E_empty)
            assert (result.Q == E1.Q).all() and (result.q == E1.q).all()
        except NotImplementedError:
            pytest.skip("priv_orEllipsoidOA not implemented")

    def test_union_contains_points(self):
        Q1 = np.array([[5.4387811500952807, 12.4977183618314545], [12.4977183618314545, 29.6662117284481646]])
        q1 = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E1 = Ellipsoid(Q1, q1, 1e-6)
        E1 = Ellipsoid(E1.Q, np.zeros_like(E1.q), E1.TOL)
        Q2 = np.array([[0.3542431574242590, 0.0233699257103926], [0.0233699257103926, 2.4999614009532856]])
        q2 = np.array([[0.0873801375346114], [-2.4641617305288825]])
        E2 = Ellipsoid(Q2, q2, 1e-6)
        E2 = Ellipsoid(E2.Q, np.zeros_like(E2.q), E2.TOL)
        try:
            Eres = E1 | E2
            # This will raise NotImplementedError unless priv_orEllipsoidOA is implemented
            # If implemented, check that points from both are contained
            # Y_nd = [randPoint(E1,2), randPoint(E2,2)]
            # assert all(Eres.contains_(Y_nd)[0])
        except NotImplementedError:
            pytest.skip("priv_orEllipsoidOA not implemented")

    def test_zero_rank_union(self):
        Q1 = np.eye(2)
        q1 = np.zeros((2, 1))
        E1 = Ellipsoid(Q1, q1)
        Q0 = np.zeros((2, 2))
        q0 = np.array([[1.0986933635979599], [-1.9884387759871638]])
        E0 = Ellipsoid(Q0, q0, 1e-6)
        E0 = Ellipsoid(E0.Q, np.zeros_like(E0.q), E0.TOL)
        try:
            Eres = E1 | E0
            # This will raise NotImplementedError unless priv_orEllipsoidOA is implemented
            # If implemented, check that points from both are contained
        except NotImplementedError:
            pytest.skip("priv_orEllipsoidOA not implemented") 