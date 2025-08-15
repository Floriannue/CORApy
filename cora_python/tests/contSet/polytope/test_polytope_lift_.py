import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_lift_basic():
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]])
    b = np.array([[1], [1], [1], [1], [1]])
    P = Polytope(A, b)
    P_high = Polytope.lift_(P, 4, [4, 2])

    # Check that constraints map to dims [4,2]
    assert P_high.A.shape[1] == 4
    # Original A columns placed in 4 and 2
    A_expected = np.zeros((A.shape[0], 4))
    A_expected[:, [3, 1]] = A
    assert np.allclose(P_high.A, A_expected)
    assert np.allclose(P_high.b, b)


