import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_convHull_two_segments():
    # Segment along x: { (t,0) | t in [0,1] }
    A1 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b1 = np.array([[1.0], [0.0], [0.0], [0.0]])
    P1 = Polytope(A1, b1)

    # Segment along y: { (0,s) | s in [0,1] }
    A2 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b2 = np.array([[0.0], [0.0], [1.0], [0.0]])
    P2 = Polytope(A2, b2)

    P = Polytope.convHull_(P1, P2)
    # Convex hull should be the triangle with vertices (0,0), (1,0), (0,1)
    V = P.V
    # Expect convex hull extreme vertices subset of expected triangle vertices
    expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # For each column in V, verify it matches one of expected vertices
    ok = True
    for j in range(V.shape[1]):
        v = V[:, j:j+1]
        matches = np.any(np.all(np.isclose(expected, v, atol=1e-10), axis=0))
        ok = ok and matches
    assert ok


