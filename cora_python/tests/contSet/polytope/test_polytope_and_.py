import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_and_intersection():
    # Square [-1,1]^2 intersect with halfspace x <= 0
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    P1 = Polytope(A, b)
    P2 = Polytope(np.array([[1.0, 0.0]]), np.array([[0.0]]))
    P = Polytope.and_(P1, P2)
    # Check representative points
    pts = np.array([[-1, -1, 0, 0], [-1, 1, -1, 1]])
    for i in range(pts.shape[1]):
        assert P.contains_(pts[:, i])


def test_polytope_and_empty_result():
    # Two disjoint halfspaces in 1D: x <= 0 and x >= 1
    P1 = Polytope(np.array([[1.0]]), np.array([[0.0]]))
    P2 = Polytope(np.array([[-1.0]]), np.array([[-1.0]]))
    P = Polytope.and_(P1, P2)
    assert P.representsa_('emptySet', 1e-10)


def test_polytope_and_equalities_stack():
    # Intersection should keep equalities
    # P1: x=1 in 1D, P2: x<=2
    Ae1 = np.array([[1.0]])
    be1 = np.array([[1.0]])
    P1 = Polytope(np.zeros((0, 1)), np.zeros((0, 1)), Ae1, be1)
    P2 = Polytope(np.array([[1.0]]), np.array([[2.0]]))
    P = Polytope.and_(P1, P2)
    # Should contain x=1, but not x=0
    r1, _, _ = P.contains_(np.array([1.0]))
    r0, _, _ = P.contains_(np.array([0.0]))
    assert bool(np.atleast_1d(r1)[0])
    assert not bool(np.atleast_1d(r0)[0])


