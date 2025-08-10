import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_minkDiffEllipsoid import priv_minkDiffEllipsoid


def test_priv_minkDiffEllipsoid_basic():
    E1 = Ellipsoid(np.array([[2.0,0.0],[0.0,1.0]]), np.zeros((2,1)))
    E2 = Ellipsoid(np.array([[1.0,0.0],[0.0,0.5]]), np.zeros((2,1)))
    L = np.zeros((2,0))
    out = priv_minkDiffEllipsoid(E1, E2, L, 'outer')
    assert isinstance(out, Ellipsoid)


