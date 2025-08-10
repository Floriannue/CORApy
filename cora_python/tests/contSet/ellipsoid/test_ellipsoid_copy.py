import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid


def test_copy_returns_equal_but_distinct():
    Q = np.array([[2.0, 0.1], [0.1, 1.0]])
    q = np.array([[0.3], [0.2]])
    E = Ellipsoid(Q, q)
    Ec = E.copy()
    assert isinstance(Ec, Ellipsoid)
    assert Ec is not E
    assert np.allclose(Ec.Q, E.Q)
    assert np.allclose(Ec.q, E.q)

