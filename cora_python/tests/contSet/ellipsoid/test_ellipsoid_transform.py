import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid


def test_transform_linear():
    Q = np.array([[2.0, 0.0],[0.0, 1.0]])
    q = np.array([[1.0],[2.0]])
    E = Ellipsoid(Q, q)
    A = np.array([[1.0, 1.0],[0.0, 2.0]])
    Et = E.transform(A)
    assert isinstance(Et, Ellipsoid)
    assert Et.Q.shape == (2,2)

