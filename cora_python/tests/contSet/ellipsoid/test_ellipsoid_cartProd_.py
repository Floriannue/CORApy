import numpy as np
import pytest

from cora_python.contSet.ellipsoid import Ellipsoid


def test_cartProd_ellipsoid_ellipsoid_via_interval():
    Q1 = np.array([[3.0, -1.0], [-1.0, 1.0]])
    q1 = np.array([[1.0], [0.0]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[5.0, 1.0], [1.0, 2.0]])
    q2 = np.array([[1.0], [-1.0]])
    E2 = Ellipsoid(Q2, q2)

    E = E1.cartProd_(E2, 'outer')

    # Result should be 4D ellipsoid with block-diagonal Q and stacked q
    assert isinstance(E, Ellipsoid)
    assert E.Q.shape == (4, 4)
    assert E.q.shape == (4, 1)


def test_cartProd_ellipsoid_numeric_column():
    Q = np.array([[2.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.5], [-0.5]])
    E = Ellipsoid(Q, q)

    s = np.array([[2.0], [3.0]])

    E_out = E.cartProd_(s, 'outer')
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (4, 4)
    assert E_out.q.shape == (4, 1)
    # numeric part adds directly to center
    assert np.allclose(E_out.q[:2], q)
    assert np.allclose(E_out.q[2:], s)


def test_cartProd_numeric_ellipsoid_column():
    s = np.array([[2.0], [3.0]])

    Q = np.array([[2.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.5], [-0.5]])
    E = Ellipsoid(Q, q)

    E_out = Ellipsoid.cartProd_(s, E, 'outer')
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (4, 4)
    assert E_out.q.shape == (4, 1)
    assert np.allclose(E_out.q[:2], s)
    assert np.allclose(E_out.q[2:], q)


def test_cartProd_wrong_mode_raises():
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    with pytest.raises(Exception):
        E.cartProd_(E, 'inner')

