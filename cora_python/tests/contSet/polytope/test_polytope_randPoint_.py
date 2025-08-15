import numpy as np

from cora_python.contSet.polytope import Polytope


def test_polytope_randPoint_standard_and_extreme():
    # Triangle
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    P = Polytope(V)

    pts_std = Polytope.randPoint_(P, 10, 'standard')
    assert pts_std.shape == (2, 10)

    pts_ext = Polytope.randPoint_(P, 10, 'extreme')
    assert pts_ext.shape == (2, 10)

    # 'all' returns vertices
    all_pts = Polytope.randPoint_(P, 'all', 'extreme')
    assert all_pts.shape[1] >= 3


def test_polytope_randPoint_fullspace_and_empty():
    P_inf = Polytope.Inf(3)
    pts = Polytope.randPoint_(P_inf, 5)
    assert pts.shape == (3, 5)

    P_empty = Polytope.empty(2)
    pts_empty = Polytope.randPoint_(P_empty, 7)
    assert pts_empty.shape == (2, 0)


