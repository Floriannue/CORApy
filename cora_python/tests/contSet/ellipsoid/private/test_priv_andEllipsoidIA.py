import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_andEllipsoidIA import priv_andEllipsoidIA


def test_priv_andEllipsoidIA_concentric_equal():
    # identical ellipsoids -> IA returns the same
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E = priv_andEllipsoidIA([E1, E2])
    assert np.allclose(E.Q, E1.Q) and np.allclose(E.q, E1.q)


def test_priv_andEllipsoidIA_nested_concentric():
    # E2 inside E1 (concentric) -> IA should be close to E2
    E1 = Ellipsoid(2*np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(0.5*np.eye(2), np.zeros((2,1)))
    E = priv_andEllipsoidIA([E1, E2])
    # Check inclusion proxy: det(Q) should not exceed E1 and should be >= E2
    assert np.linalg.det(E.Q) <= np.linalg.det(E1.Q) + 1e-6
    assert np.linalg.det(E.Q) >= np.linalg.det(E2.Q) - 1e-6


