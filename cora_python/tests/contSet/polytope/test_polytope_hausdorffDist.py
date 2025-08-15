import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_hausdorff_bounded_sets():
    # Unit square and shifted unit square
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [0], [1], [0]])
    P = Polytope(A, b)
    # Shift P by t=(0,0.5): new b' = b + A*t
    t = np.array([[0.0], [0.5]])
    Q = Polytope(A, b + A @ t)

    d = Polytope.hausdorffDist(P, Q)
    # Expected Hausdorff distance is 0.5 (vertical shift)
    assert np.isclose(d, 0.5, atol=1e-6)


def test_polytope_hausdorff_empty_and_unbounded():
    P_empty = Polytope.empty(2)
    P_box = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([[1], [1], [1], [1]]))
    P_inf = Polytope.Inf(2)

    assert np.isinf(Polytope.hausdorffDist(P_empty, P_box))
    assert np.isinf(Polytope.hausdorffDist(P_inf, P_box))


