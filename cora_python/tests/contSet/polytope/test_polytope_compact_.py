import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_compact_redundant_inequalities():
    # Square with duplicated constraints
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                  [1, 0], [0, 1]])
    b = np.array([[1], [1], [1], [1], [1], [1]])
    P = Polytope(A, b)
    P2 = Polytope.compact_(P)
    # After compaction, constraints should be fewer or equal, and set identical
    assert P2.A.shape[0] <= A.shape[0]
    # Pick test points to ensure identity
    pts = np.array([[0, 1, 0, -1], [0, 0, 1, -1]])
    for i in range(pts.shape[1]):
        r1, _, _ = P.contains_(pts[:, i])
        r2, _, _ = P2.contains_(pts[:, i])
        assert bool(np.atleast_1d(r1)[0]) == bool(np.atleast_1d(r2)[0])


def test_polytope_compact_vertices_hull():
    # V-rep with redundant interior point
    V = np.array([[0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.2]])
    P = Polytope(V)
    P2 = Polytope.compact_(P)
    # Expect only the triangle vertices remain
    assert P2.V.shape[1] == 3


