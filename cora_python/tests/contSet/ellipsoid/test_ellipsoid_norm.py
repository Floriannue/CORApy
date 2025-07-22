import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

class TestEllipsoidNorm:
    def test_empty_case(self):
        n = 2
        E = Ellipsoid.empty(n)
        assert E.norm_() == -np.inf

    def test_2d_cases(self):
        # E1
        Q1 = np.array([[5.4387811500952807, 12.4977183618314545], [12.4977183618314545, 29.6662117284481646]])
        q1 = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E1 = Ellipsoid(Q1, q1, 1e-6)
        E1 = Ellipsoid(E1.Q, np.zeros_like(E1.q), E1.TOL)  # remove center
        # Ed1
        Qd1 = np.array([[4.2533342807136076, 0.6346400221575308], [0.6346400221575309, 0.0946946398147988]])
        qd1 = np.array([[-2.4653656883489115], [0.2717868749873985]])
        Ed1 = Ellipsoid(Qd1, qd1, 1e-6)
        Ed1 = Ellipsoid(Ed1.Q, np.zeros_like(Ed1.q), Ed1.TOL)
        # E0
        Q0 = np.zeros((2, 2))
        q0 = np.array([[1.0986933635979599], [-1.9884387759871638]])
        E0 = Ellipsoid(Q0, q0, 1e-6)
        E0 = Ellipsoid(E0.Q, np.zeros_like(E0.q), E0.TOL)
        # Check norm_ does not error and is non-negative
        assert E1.norm_() >= 0
        assert Ed1.norm_() >= 0
        assert E0.norm_() >= 0

    def test_nonzero_center(self):
        Q = np.eye(2)
        q = np.array([[1.0], [0.0]])
        E = Ellipsoid(Q, q)
        with pytest.raises(Exception):
            E.norm_()

    def test_wrong_type(self):
        Q = np.eye(2)
        E = Ellipsoid(Q, np.zeros((2, 1)))
        with pytest.raises(Exception):
            E.norm_(1)
        with pytest.raises(Exception):
            E.norm_('fro') 