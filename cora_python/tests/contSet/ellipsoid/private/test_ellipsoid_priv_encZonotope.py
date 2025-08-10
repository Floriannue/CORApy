import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_encZonotope import priv_encZonotope


def test_priv_encZonotope_basic():
    E = Ellipsoid(np.array([[2.0,0.0],[0.0,1.0]]), np.zeros((2,1)))
    Z = priv_encZonotope(E, 6)
    assert hasattr(Z, 'c') and hasattr(Z, 'G')


