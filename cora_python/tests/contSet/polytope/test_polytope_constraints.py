import numpy as np
from cora_python.contSet.polytope.polytope import Polytope


def test_reps_both_on_after_conversions():
    # Start from V-rep, convert to H-rep
    V = np.array([[0, 1, 0], [0, 0, 1]])
    P = Polytope(V)
    assert P.isVRep and not P.isHRep
    P.constraints()
    assert P.isHRep  # H-rep computed
    # Now compute vertices from H-rep (still bounded)
    P.vertices_()
    # MATLAB behavior: both reps can be present
    assert P.isVRep and P.isHRep

def test_constraints_single_vertex_2d():
    V = np.array([[2], [3]])  # single point in 2D
    P = Polytope(V)
    P.constraints()
    # For single vertex: A empty, Ae = I, be = V
    assert P.A.shape == (0, 2)
    assert P.Ae.shape == (2, 2)
    assert np.allclose(P.be.flatten(), V.flatten())
    # Both reps should remain available
    assert P.isVRep and P.isHRep


def test_constraints_1d_bounded_segment():
    V = np.array([[ -1.0, 2.0 ]])  # 1 x 2 vertices
    P = Polytope(V)
    P.constraints()
    # Expect A = [[1],[-1]], b = [[max],[ -min ]]
    assert P.A.shape == (2, 1)
    assert np.allclose(sorted(P.b.flatten().tolist()), sorted([2.0, 1.0]))
    # Check that both vertices satisfy A x <= b with equality
    for x in V.flatten():
        val = (P.A @ np.array([[x]])).flatten()
        assert np.all(val <= P.b.flatten() + 1e-12)


def test_constraints_2d_degenerate_line():
    V = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])  # collinear along y=x
    P = Polytope(V)
    P.constraints()
    # Degenerate line: expect 4 inequalities, no equalities
    assert P.A.shape == (4, 2)
    assert P.Ae.shape == (0, 2)
    # All points satisfy A x <= b
    for i in range(V.shape[1]):
        x = V[:, i:i+1]
        assert np.all((P.A @ x).flatten() <= P.b.flatten() + 1e-10)


def test_constraints_2d_square_from_vertices():
    V = np.array([[1, -1, -1, 1], [1, 1, -1, -1]])  # square
    P = Polytope(V)
    P.constraints()
    # Each vertex lies on at least one facet (A x == b within tol)
    satisfied = []
    for i in range(V.shape[1]):
        x = V[:, i:i+1]
        vals = (P.A @ x).flatten()
        eq = np.any(np.isclose(vals, P.b.flatten(), atol=1e-9))
        satisfied.append(eq)
        assert np.all(vals <= P.b.flatten() + 1e-9)
    assert all(satisfied)


def test_constraints_hrep_no_change_flags():
    # Build H-rep unit square
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.ones((4, 1))
    P = Polytope(A, b)
    assert P.isHRep and not P.isVRep
    P.constraints()
    # Stays H-rep; V-rep unchanged
    assert P.isHRep
    # A, b should be consistent shapes
    assert P.A.shape == (4, 2)
    assert P.b.shape == (4, 1)
