import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_containsEllipsoid import priv_containsEllipsoid


def test_priv_containsEllipsoid_basic():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(0.5*np.eye(2), np.zeros((2,1)))
    res, cert, scaling = priv_containsEllipsoid(E1, E2, 1e-10)
    assert isinstance(res, (bool, np.bool_))


