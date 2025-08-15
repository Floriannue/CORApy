import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_volume_box_matches_product():
    # Box [0,1] x [0,2]
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [0], [2], [0]])
    P = Polytope(A, b)
    vol = P.volume_()
    assert np.isclose(vol, 2.0, atol=1e-10)


def test_polytope_volume_unbounded_inf():
    # Half-space x <= 1 in 2D
    P = Polytope(np.array([[1.0, 0.0]]), np.array([[1.0]]))
    vol = P.volume_()
    assert np.isinf(vol)


def test_polytope_volume_degenerate_zero():
    # Line segment on x-axis from 0 to 1 in 2D (degenerate area -> 0)
    Ae = np.array([[0.0, 1.0]])
    be = np.array([[0.0]])
    A = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([[1.0], [0.0]])
    P = Polytope(A, b, Ae, be)
    vol = P.volume_()
    assert np.isclose(vol, 0.0, atol=1e-12)


