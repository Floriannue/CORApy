import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_or_convex_hull_union():
    # Two points at origin and (1,1) via H-rep boxes
    P = Polytope(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), np.array([[0.0], [0.0], [0.0], [0.0]]))
    Q = Polytope(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), np.array([[1.0], [-1.0], [1.0], [-1.0]]))
    H = Polytope.or_(P, Q)
    assert H.V.shape[1] >= 2


