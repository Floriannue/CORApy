import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_inscParallelotope import priv_inscParallelotope


def test_priv_inscParallelotope_basic():
    E = Ellipsoid(np.array([[3.0,0.0],[0.0,1.0]]), np.zeros((2,1)))
    Z = priv_inscParallelotope(E)
    assert hasattr(Z, 'c') and hasattr(Z, 'G')


