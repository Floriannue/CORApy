import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_boundary import priv_boundary


def test_priv_boundary_counts_and_dims():
    E = Ellipsoid(np.array([[2.0, 0.0], [0.0, 1.0]]), np.zeros((2, 1)))
    Y, L = priv_boundary(E, 50)
    assert Y.shape[1] == 50
    assert Y.shape[0] == 2 and L.shape[0] == 2


