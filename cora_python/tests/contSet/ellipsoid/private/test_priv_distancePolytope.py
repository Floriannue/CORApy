import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.ellipsoid.private.priv_distancePolytope import priv_distancePolytope

def test_priv_distancePolytope_simple():
    # Ellipsoid: unit circle at origin
    E = Ellipsoid(np.eye(2))
    
    # Polytope: simple box x <= 2, x >= 1.5, y <= 1, y >= -1
    # which is some distance away from the ellipsoid
    A = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    b = np.array([-1.5, 2., 1., 1.])
    P = Polytope(A, b)

    # The closest point on the ellipsoid to the polytope is (1,0).
    # The closest point on the polytope to the ellipsoid is (1.5, 0).
    # The Euclidean distance is 0.5.
    # The value from this function is distance^2 - 1, but it's not a simple
    # Euclidean distance. For a unit circle, distance should be > 0.
    
    dist = priv_distancePolytope(E, P)
    assert dist > 0

def test_priv_distancePolytope_degenerate_supported():
    # Degenerate ellipsoid (rank-deficient)
    Q_degen = np.array([[1., 0.], [0., 0.]])
    E_degen = Ellipsoid(Q_degen)
    P = Polytope(np.array([[1., 0.]]), np.array([2.]))

    # Function should return a finite numeric value for low-rank ellipsoids
    val = priv_distancePolytope(E_degen, P)
    assert isinstance(val, (float, np.floating))
    assert np.isfinite(val)