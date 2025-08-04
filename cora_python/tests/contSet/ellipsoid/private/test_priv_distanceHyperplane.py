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
    P_sep = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1.0, 0.0]]), np.array([[3.0]]))

    # Center is at x=1. Ellipsoid radius is 1. Edge of E is at x=2.
    # Distance from edge of E (x=2) to hyperplane (x=3) is 1.
    dist_sep = priv_distanceHyperplane(E, P_sep)
    assert np.isclose(dist_sep, 1.0, atol=1e-9)

    # Case 2: Intersecting
    # Hyperplane x=1 -> c=[1,0], y=1 (passes through center)
    P_int = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1.0, 0.0]]), np.array([[1.0]]))
    # Distance from center to hyperplane is 0. Distance from ellipsoid edge to hyperplane is -1.
    dist_int = priv_distanceHyperplane(E, P_int)
    assert np.isclose(dist_int, -1.0, atol=1e-9)

    # Case 3: Contained
    # Hyperplane x=0 -> c=[1,0], y=0 (E is in x>0, so it's on one side)
    P_cont = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.array([[1.0, 0.0]]), np.array([[0.0]]))
    # Distance from ellipsoid edge (x=0) to hyperplane (x=0) is 0.
    dist_cont = priv_distanceHyperplane(E, P_cont)
    assert np.isclose(dist_cont, 0.0, atol=1e-9)

    # Case 4: Hyperplane with zero normal vector c, and y=0
    # Equation 0=0 -> whole space. Distance should be -inf.
    P_degen1 = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.zeros((1,2)), np.array([[0.0]]))
    dist_degen1 = priv_distanceHyperplane(E, P_degen1)
    assert dist_degen1 == -np.inf

    # Case 5: Hyperplane with zero normal vector c, and y!=0
    # Equation 0=1 -> empty set. Distance should be +inf.
    P_degen2 = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), np.zeros((1,2)), np.array([[1.0]]))
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
    hyp1 = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), v.T, v.T @ s)
    assert priv_distanceHyperplane(E, hyp1) <= E.TOL

    # 2. Not intersecting
    # s is a point outside the ellipsoid
    s = E.q + 2.0 * (E.Q @ v) # Point outside
    hyp2 = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), v.T, v.T @ s)
    assert priv_distanceHyperplane(E, hyp2) > E.TOL

    # 3. Guaranteed to not touch or intersect
    val = E.supportFunc_(v, 'upper') # Calculate the support function value
    hyp3 = Polytope(np.array([[]]).reshape(0,2), np.array([[]]).reshape(0,1), v.T, np.array([[val[0] + 0.1]]))
    assert priv_distanceHyperplane(E, hyp3) > E.TOL 