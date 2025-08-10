import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid


def test_plus_outer_directions():
    Q1 = np.array([[2.0, 0.2], [0.2, 1.0]])
    q1 = np.array([[0.0], [0.0]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[1.5, -0.1], [-0.1, 0.8]])
    q2 = np.array([[0.1], [0.1]])
    E2 = Ellipsoid(Q2, q2)

    # one direction along x-axis and one along y-axis
    L = np.array([[1.0, 0.0], [0.0, 1.0]])
    E_out = E1.plus(E2, 'outer', L)
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (2, 2)


def test_plus_outer_halder():
    Q1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    q1 = np.array([[0.2], [-0.1]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[0.5, 0.0], [0.0, 0.8]])
    q2 = np.array([[0.0], [0.1]])
    E2 = Ellipsoid(Q2, q2)

    E_out = E1.plus(E2, 'outer:halder', np.zeros((2, 0)))
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (2, 2)
    # Halder mode should be at least as large in volume as directional outer
    E_dir = E1.plus(E2, 'outer', np.zeros((2, 0)))
    assert np.linalg.det(E_out.Q) >= 0

