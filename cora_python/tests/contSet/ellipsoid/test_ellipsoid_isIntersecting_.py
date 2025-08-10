import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid


def test_isIntersecting_point_inside():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.0], [0.0]])
    E = Ellipsoid(Q, q)
    p = np.array([[0.1], [0.1]])
    assert E.isIntersecting_(p, 'exact', 1e-9)

