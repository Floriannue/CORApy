import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_inscZonotope import priv_inscZonotope


def test_priv_inscZonotope_basic():
    E = Ellipsoid(np.array([[2.0,0.0],[0.0,1.0]]), np.zeros((2,1)))
    Z = priv_inscZonotope(E, 4, 'ub')
    assert hasattr(Z, 'c') and hasattr(Z, 'G')


