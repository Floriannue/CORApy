import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_isIntersectingMixed import priv_isIntersectingMixed
from cora_python.contSet.zonotope import Zonotope


def test_priv_isIntersectingMixed_basic():
    E = Ellipsoid(np.eye(2), np.zeros((2,1)))
    Z = Zonotope(np.zeros((2,1)), 0.1*np.eye(2))
    res = priv_isIntersectingMixed(E, Z)
    assert isinstance(res, (bool, np.bool_))


