import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_cartProd_basic():
    # P1: 1D interval [0,1]
    A1 = np.array([[1.0], [-1.0]])
    b1 = np.array([[1.0], [0.0]])
    P1 = Polytope(A1, b1)

    # P2: 1D interval [-2, 3]
    A2 = np.array([[1.0], [-1.0]])
    b2 = np.array([[3.0], [2.0]])
    P2 = Polytope(A2, b2)

    P = Polytope.cartProd_(P1, P2)
    assert P.dim() == 2
    # Check that corners are contained
    corners = np.array([[0, 0, 1, 1], [-2, 3, -2, 3]])
    for i in range(corners.shape[1]):
        assert P.contains_(corners[:, i])


def test_polytope_cartProd_with_point():
    # P1: point at x=2
    Ae1 = np.array([[1.0]])
    be1 = np.array([[2.0]])
    P1 = Polytope(np.zeros((0, 1)), np.zeros((0, 1)), Ae1, be1)

    # P2: 1D interval [-1, 1]
    A2 = np.array([[1.0], [-1.0]])
    b2 = np.array([[1.0], [1.0]])
    P2 = Polytope(A2, b2)

    P = Polytope.cartProd_(P1, P2)
    assert P.dim() == 2
    # Check that (2, y) for y in [-1,1] are contained
    ys = [-1.0, 0.0, 1.0]
    for y in ys:
        assert P.contains_(np.array([2.0, y]))


def test_polytope_cartProd_empty_operand():
    # Empty in 2D
    E = Polytope.empty(2)
    # Box in 2D
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    B = Polytope(A, b)

    P = Polytope.cartProd_(E, B)
    assert P.representsa_('emptySet', 1e-10)


