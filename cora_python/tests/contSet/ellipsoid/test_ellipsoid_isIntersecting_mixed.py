import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.zonotope import Zonotope


def test_isIntersecting_mixed_with_zonotope_quadmap():
    Q = np.array([[2.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.0], [0.0]])
    E = Ellipsoid(Q, q)

    c = np.array([[0.0], [0.0]])
    G = np.array([[0.2, 0.0], [0.0, 0.2]])
    Z = Zonotope(c, G)

    res = E.isIntersecting_(Z, 'approx', 1e-6)
    assert isinstance(res, (bool, np.bool_))

