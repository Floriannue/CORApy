import numpy as np

from cora_python.contSet.polytope import Polytope
from cora_python.contSet.interval import Interval


def test_polytope_reduceOverDomain_basic_outer_inner():
    # Build a set of similar halfspaces around x <= 1 in 1D
    A = np.array([[1.0], [0.9], [1.1]])
    b = np.array([[1.0], [1.0], [1.0]])
    P = Polytope(A, b)

    dom = Interval(np.array([[0.0]]), np.array([[1.0]]))

    P_out = Polytope.reduceOverDomain(P, dom, 'outer')
    P_in = Polytope.reduceOverDomain(P, dom, 'inner')

    # Outer should contain all originals over dom
    # Test a grid of points in dom
    xs = np.linspace(0.0, 1.0, 5).reshape(1, -1)
    for x in xs.T:
        r_out, _, _ = P_out.contains_(x)
        r_all = True
        for i in range(A.shape[0]):
            r_all = r_all and (A[i, 0] * x[0] <= b[i, 0] + 1e-10)
        assert bool(r_out) >= bool(r_all)

    # Inner should be contained in intersection of originals
    for x in xs.T:
        r_in, _, _ = P_in.contains_(x)
        r_inter = True
        for i in range(A.shape[0]):
            r_inter = r_inter and (A[i, 0] * x[0] <= b[i, 0] + 1e-10)
        assert (not r_in) or r_inter


