import numpy as np
import pytest

from cora_python.contSet.ellipsoid import Ellipsoid


def test_polygon_2d():
    Q = np.array([[2.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.0], [0.0]])
    E = Ellipsoid(Q, q)
    pg = E.polygon()
    # polygon(V) should produce an object; basic sanity
    assert hasattr(pg, 'vertices') or hasattr(pg, 'V')


def test_polygon_wrong_dim_raises():
    Q = np.eye(3)
    q = np.zeros((3, 1))
    E = Ellipsoid(Q, q)
    with pytest.raises(Exception):
        E.polygon()

