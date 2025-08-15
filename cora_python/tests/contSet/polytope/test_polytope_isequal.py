import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_isequal_same_box():
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    P = Polytope(A, b)
    Q = Polytope(A.copy(), b.copy())
    assert Polytope.isequal(P, Q)


def test_polytope_isequal_redundant_constraints():
    A1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b1 = np.array([[1], [1], [1], [1]])
    P = Polytope(A1, b1)
    A2 = np.vstack([A1, np.array([[1, 0]])])
    b2 = np.vstack([b1, np.array([[1]])])
    Q = Polytope(A2, b2)
    assert Polytope.isequal(P, Q)


def test_polytope_isequal_different_sets():
    P = Polytope(np.array([[1.0]]), np.array([[1.0]]))
    Q = Polytope(np.array([[1.0]]), np.array([[2.0]]))
    assert not Polytope.isequal(P, Q)


