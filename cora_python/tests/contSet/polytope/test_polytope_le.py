import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_le_subset():
    # P: [-1,1] in 1D, Q: [-2,2] in 1D -> P <= Q
    P = Polytope(np.array([[1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    Q = Polytope(np.array([[1.0], [-1.0]]), np.array([[2.0], [2.0]]))
    assert Polytope.le(P, Q)
    assert not Polytope.le(Q, P)


