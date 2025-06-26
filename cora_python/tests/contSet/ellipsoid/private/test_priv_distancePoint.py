import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_distancePoint import priv_distancePoint as priv_distancePoint_func

# Attach the private method for testing
Ellipsoid.priv_distancePoint = priv_distancePoint_func

def test_priv_distancePoint_full_dim():
    E = Ellipsoid(np.eye(2), np.zeros((2,1)))
    
    # Point inside
    p_in = np.array([[0.5], [0]])
    dist_in = E.priv_distancePoint(p_in)
    assert dist_in < 0

    # Point on boundary
    p_on = np.array([[1], [0]])
    dist_on = E.priv_distancePoint(p_on)
    assert np.isclose(dist_on, 0)

    # Point outside
    p_out = np.array([[2], [1]])
    dist_out = E.priv_distancePoint(p_out)
    assert dist_out > 0

    # Test with multiple points
    points = np.hstack([p_in, p_on, p_out])
    dists = E.priv_distancePoint(points)
    assert dists.shape == (3,)
    assert dists[0] < 0
    assert np.isclose(dists[1], 0)
    assert dists[2] > 0

def test_priv_distancePoint_degenerate():
    # Degenerate ellipsoid
    Q = np.array([[4.0, 0.0], [0.0, 0.0]])
    q = np.array([[1.0], [1.0]])
    E = Ellipsoid(Q, q)
    
    # Point differing only in non-degenerate dim, on boundary
    p1 = np.array([[3.0], [1.0]]) # (x-q) = [2,0]. dist_nd = (2^2)/4 -1 = 0
    dist1 = E.priv_distancePoint(p1)
    assert np.isclose(dist1, 0)
    
    # Point differing in degenerate dim
    p2 = np.array([[1.0], [2.0]]) # (x-q) = [0,1]. dist_nd=0. dist_d = ((1/sqrt(TOL))^2) - 1 > 0
    dist2 = E.priv_distancePoint(p2)
    assert dist2 > 0

    # Point differing in both
    p3 = np.array([[3.0], [2.0]])
    dist3 = E.priv_distancePoint(p3)
    assert dist3 > 0
    
    # Point is the center
    dist4 = E.priv_distancePoint(E.q)
    assert dist4 < 0 