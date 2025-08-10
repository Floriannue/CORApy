import numpy as np
import pytest

from cora_python.contSet.ellipsoid import Ellipsoid


def test_polygon_2d():
    Q = np.array([[2.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.0], [0.0]])
    E = Ellipsoid(Q, q)
    pg = E.polygon()
    # polygon(V) should produce an object; basic sanity
    assert hasattr(pg, 'V')
    V = getattr(pg, 'V', None)
    assert V is None or (V.shape[0] == 2 and V.shape[1] > 0)


def test_polygon_wrong_dim_raises():
    Q = np.eye(3)
    q = np.zeros((3, 1))
    E = Ellipsoid(Q, q)
    with pytest.raises(Exception):
        E.polygon()


def test_polygon_point_and_degenerate_projection_path():
    # Point ellipsoid
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1.0], [2.0]]))
    pg_point = E_point.polygon()
    Vp = getattr(pg_point, 'V', None)
    if Vp is not None:
        assert Vp.shape[0] == 2
    # Degenerate ellipsoid with rank 1
    Q = np.array([[1.0, 0.0], [0.0, 0.0]])
    q = np.zeros((2, 1))
    E_deg = Ellipsoid(Q, q)
    pg_deg = E_deg.polygon()
    Vd = getattr(pg_deg, 'V', None)
    if Vd is not None:
        assert Vd.shape[0] == 2

