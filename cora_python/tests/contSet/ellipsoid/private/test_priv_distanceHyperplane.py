import numpy as np
import pytest

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.ellipsoid.private.priv_distanceHyperplane import priv_distanceHyperplane

def test_priv_distanceHyperplane():
    # Ellipsoid: (x-q)'Q^{-1}(x-q) <= 1
    Q = np.eye(2)
    q = np.array([[1], [0]])
    E = Ellipsoid(Q, q)

    # Hyperplane: c'x = y
    
    # Case 1: Separated
    # Hyperplane x=3 -> c=[1,0], y=3
    P_sep = Polytope(np.array([[1.0, 0.0]]), np.array([[3.0]]))
    # Center is at x=1. Ellipsoid radius is 1. Edge of E is at x=2.
    # Distance from edge of E (x=2) to hyperplane (x=3) is 1.
    dist_sep = priv_distanceHyperplane(E, P_sep)
    assert np.isclose(dist_sep, 1.0)

    # Case 2: Touching
    # Hyperplane x=2 -> c=[1,0], y=2
    P_touch = Polytope(np.array([[1.0, 0.0]]), np.array([[2.0]]))
    dist_touch = priv_distanceHyperplane(E, P_touch)
    assert np.isclose(dist_touch, 0.0)

    # Case 3: Intersecting
    # Hyperplane x=1.5 -> c=[1,0], y=1.5
    P_inter = Polytope(np.array([[1.0, 0.0]]), np.array([[1.5]]))
    dist_inter = priv_distanceHyperplane(E, P_inter)
    assert dist_inter < 0

    # Case 4: Hyperplane with zero normal vector c, and y=0
    # Equation 0=0 -> whole space. Distance should be -inf.
    P_degen1 = Polytope(np.zeros((1,2)), np.array([0.0]))
    dist_degen1 = priv_distanceHyperplane(E, P_degen1)
    assert dist_degen1 == -np.inf

    # Case 5: Hyperplane with zero normal vector c, and y!=0
    # Equation 0=1 -> empty set. Distance should be +inf.
    P_degen2 = Polytope(np.zeros((1,2)), np.array([1.0]))
    dist_degen2 = priv_distanceHyperplane(E, P_degen2)
    assert dist_degen2 == np.inf

def test_priv_distanceHyperplane_from_matlab():
    # Based on aux_distanceHyperplane from testLong_ellipsoid_distance.m
    
    # Ellipsoid centered at origin
    E = Ellipsoid(np.diag([4, 1]), np.zeros((2,1)))
    
    # Direction
    v = np.array([[1], [1]])
    v = v / np.linalg.norm(v)

    # 1. Guaranteed to intersect
    # s is a point inside the ellipsoid
    s = E.q + 0.5 * (E.Q @ v) # Just some internal point
    hyp1 = Polytope(v.T, v.T @ s)
    assert priv_distanceHyperplane(E, hyp1) <= E.TOL

    # 2. Guaranteed to touch
    val, _ = E.supportFunc_(v, 'upper')
    hyp2 = Polytope(v.T, np.array([val]))
    assert np.isclose(priv_distanceHyperplane(E, hyp2), 0, atol=E.TOL)

    # 3. Guaranteed to not touch or intersect
    hyp3 = Polytope(v.T, np.array([val + 0.1]))
    assert priv_distanceHyperplane(E, hyp3) > E.TOL 