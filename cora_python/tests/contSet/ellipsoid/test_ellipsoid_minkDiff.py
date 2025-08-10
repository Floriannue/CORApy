import numpy as np
import pytest

from cora_python.contSet.ellipsoid import Ellipsoid


def test_minkDiff_numeric_vector_outer():
    Q = np.array([[2.0, 0.1], [0.1, 1.0]])
    q = np.array([[0.5], [0.0]])
    E = Ellipsoid(Q, q)
    s = np.array([[0.1], [0.2]])
    out = E.minkDiff(s, 'outer', np.zeros((2, 0)))
    assert isinstance(out, Ellipsoid)
    # center shifted by -s
    assert np.allclose(out.q, q - s)


def test_minkDiff_ellipsoid_outer_basic():
    E1 = Ellipsoid(np.array([[2.0, 0.0], [0.0, 1.5]]), np.array([[0.2], [0.1]]))
    E2 = Ellipsoid(np.array([[1.0, 0.0], [0.0, 0.5]]), np.array([[0.1], [0.0]]))
    out = E1.minkDiff(E2, 'outer', np.zeros((2, 0)))
    assert isinstance(out, Ellipsoid)
    assert out.Q.shape == (2, 2)


def test_minkDiff_ellipsoid_equal_Q():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    q1 = np.array([[0.3], [0.2]])
    q2 = np.array([[0.1], [0.0]])
    E1 = Ellipsoid(Q, q1)
    E2 = Ellipsoid(Q, q2)
    out = E1.minkDiff(E2, 'outer', np.zeros((2, 0)))
    # With equal Q, shape should be zeros
    assert np.allclose(out.Q, 0)
    assert np.allclose(out.q, q1 - q2)


def test_minkDiff_ellipsoid_not_bigger_returns_empty():
    # Construct E2 not contained in E1
    E1 = Ellipsoid(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.0], [0.0]]))
    E2 = Ellipsoid(np.array([[10.0, 0.0], [0.0, 10.0]]), np.array([[0.0], [0.0]]))
    out = E1.minkDiff(E2, 'outer', np.zeros((2, 0)))
    assert out.isemptyobject()


def test_minkDiff_list_of_ellipsoids_outer():
    E = Ellipsoid(np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([[0.3], [0.1]]))
    Elist = [Ellipsoid(np.eye(2)*0.1, np.zeros((2,1))), Ellipsoid(np.eye(2)*0.2, np.zeros((2,1)))]
    out = E.minkDiff(Elist, 'outer', np.zeros((2, 0)))
    assert isinstance(out, Ellipsoid)
    assert out.Q.shape == (2, 2)

