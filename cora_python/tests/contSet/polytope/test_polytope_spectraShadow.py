import numpy as np
import pytest

from cora_python.contSet.polytope import Polytope


def test_polytope_spectraShadow_basic_properties():
    # MATLAB example in @polytope/spectraShadow.m header
    A = np.array([[1, 2], [-1, 1], [-1, -3], [2, -1]])
    b = np.ones((4, 1))
    P = Polytope(A, b)

    SpS = P.spectraShadow()

    # Properties should mirror polytope
    assert SpS.bounded.val == P.isBounded()
    assert SpS.emptySet.val == P.representsa_('emptySet', 1e-10)
    assert SpS.fullDim.val == P.isFullDim()
    # Centers should match
    Pc = P.center()
    Sc = SpS.center.val
    Pc = Pc.reshape(-1, 1) if Pc.ndim == 1 else Pc
    Sc = Sc.reshape(-1, 1) if isinstance(Sc, np.ndarray) and Sc.ndim == 1 else Sc
    assert np.allclose(Sc.astype(float), Pc.astype(float), atol=1e-10, rtol=0)


def test_polytope_spectraShadow_fullspace_no_constraints():
    n = 3
    P = Polytope.Inf(n)

    SpS = P.spectraShadow()

    assert SpS.fullDim.val is True
    assert SpS.emptySet.val is False
    assert SpS.bounded.val is False
    # dimension via c
    assert SpS.c.shape[0] == n


def test_polytope_spectraShadow_equalities_only_point():
    # P defined by equalities x=1, y=-2
    Ae = np.eye(2)
    be = np.array([[1.0], [-2.0]])
    P = Polytope(np.zeros((0, 2)), np.zeros((0, 1)), Ae, be)

    SpS = P.spectraShadow()

    # Should be a single point -> not full-dim, bounded
    assert SpS.fullDim.val is False
    assert SpS.bounded.val is True
    assert SpS.emptySet.val is False
    assert np.allclose(SpS.center.val.reshape(-1, 1), be, atol=1e-10, rtol=0)


def test_polytope_spectraShadow_unbounded_case():
    # Half-space x <= 1 in 2D -> unbounded
    A = np.array([[1.0, 0.0]])
    b = np.array([[1.0]])
    P = Polytope(A, b)

    SpS = P.spectraShadow()

    assert SpS.bounded.val is False
    assert SpS.emptySet.val is False


def test_polytope_spectraShadow_A_shape_matches_constraints():
    # Build polytope with inequalities and equalities
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[2.0], [3.0]])
    Ae = np.array([[1.0, -1.0]])
    be = np.array([[0.0]])
    P = Polytope(A, b, Ae, be)

    # C stacks A; Ae; -Ae -> K rows
    K = A.shape[0] + 2 * Ae.shape[0]
    n = P.dim()

    SpS = P.spectraShadow()
    # A is k x (k*(n+1)) per construction
    A_mat = SpS.A
    assert A_mat.shape == (K, K * (n + 1))


