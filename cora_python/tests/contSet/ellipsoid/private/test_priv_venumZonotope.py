import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.ellipsoid.private.priv_venumZonotope import priv_venumZonotope


def test_priv_venumZonotope_basic():
    E = Ellipsoid(np.eye(2), np.zeros((2,1)))
    Z = Zonotope(np.zeros((2,1)), 0.2*np.eye(2))
    res, cert, scaling = priv_venumZonotope(E, Z, 1e-10, False)
    assert isinstance(res, (bool, np.bool_))


