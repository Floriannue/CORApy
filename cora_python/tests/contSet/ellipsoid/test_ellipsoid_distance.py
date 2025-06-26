"""
test_ellipsoid_distance - unit test function of distance

This module tests the ellipsoid distance implementation exactly matching MATLAB.

Authors:       Victor Gassmann (MATLAB), Python translation by AI Assistant
Written:       26-July-2021 (MATLAB)
Last update:   07-July-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_ellipsoid_distance():
    """Main distance test matching MATLAB test_ellipsoid_distance"""
    
    # Init cases (exact MATLAB values)
    E1 = Ellipsoid(
        np.array([[5.4387811500952807, 12.4977183618314545], 
                  [12.4977183618314545, 29.6662117284481646]]), 
        np.array([[-0.7445068341257537], [3.5800647524843665]]),
        0.000001
    )
    E0 = Ellipsoid(
        np.array([[0.0000000000000000, 0.0000000000000000], 
                  [0.0000000000000000, 0.0000000000000000]]), 
        np.array([[1.0986933635979599], [-1.9884387759871638]]),
        0.000001
    )
    
    # Check all-zero ellipsoid (E0 is degenerate, reduces to point)
    dist_ellipsoid = E1.distance(E0)
    dist_point = E1.distance(E0.q)
    assert withinTol(dist_ellipsoid, dist_point, E1.TOL), \
        f"Distance to degenerate ellipsoid {dist_ellipsoid} should equal distance to its center {dist_point}"
    
    n = len(E1.q)
    
    # Check polytope: construct hyperplane through E1.q
    l1 = np.random.randn(1, n)
    l1 = l1 / np.linalg.norm(l1)  # Normalize for numerical stability
    
    try:
        from cora_python.contSet.polytope import Polytope
        H = Polytope(l1, l1 @ E1.q)
        
        # Check if distance is <0 (hyperplane contains center of ellipsoid)
        dist_hyperplane = E1.distance(H)
        assert dist_hyperplane < 0, f"Distance to hyperplane containing center should be negative, got {dist_hyperplane}"
        
        # Check polytope: construct second hyperplane (also contains center of ellipsoid)
        l2 = np.random.randn(1, n) 
        l2 = l2 / np.linalg.norm(l2)  # Normalize for numerical stability
        
        P = Polytope(
            np.vstack([l1, l2]), 
            np.array([l1 @ E1.q, l2 @ E1.q]).flatten()
        )
        dist_polytope = E1.distance(P)
        assert dist_polytope <= 1e-6, f"Distance to polytope containing center should be near zero, got {dist_polytope}"
        
    except ImportError:
        pytest.skip("Polytope class not available for hyperplane/polytope tests")


def test_distance_point_basic():
    """Test distance to points with different positions"""
    
    # Unit ellipsoid at origin
    E = Ellipsoid(np.eye(2))
    
    # Point inside ellipsoid
    p_in = np.array([[0.5], [0]])
    dist_in = E.distance(p_in)
    assert dist_in < 0, f"Distance to interior point should be negative, got {dist_in}"
    
    # Point on boundary
    p_on = np.array([[1.0], [0]])
    dist_on = E.distance(p_on)
    assert np.abs(dist_on) < 1e-10, f"Distance to boundary point should be zero, got {dist_on}"
    
    # Point outside ellipsoid
    p_out = np.array([[2.0], [0]])
    dist_out = E.distance(p_out)
    assert dist_out > 0, f"Distance to exterior point should be positive, got {dist_out}"


def test_distance_ellipsoid_to_ellipsoid():
    """Test distance between two ellipsoids"""
    
    # Two unit ellipsoids
    E1 = Ellipsoid(np.eye(2), np.array([[0], [0]]))
    E2 = Ellipsoid(np.eye(2), np.array([[3], [0]]))
    
    # Distance should be positive (separated ellipsoids)
    dist = E1.distance(E2)
    assert dist > 0, f"Distance between separated ellipsoids should be positive, got {dist}"
    
    # Check symmetry
    dist_reverse = E2.distance(E1)
    assert np.abs(dist - dist_reverse) < 1e-10, f"Distance should be symmetric, got {dist} vs {dist_reverse}"


def test_distance_degenerate_ellipsoid():
    """Test distance involving degenerate ellipsoids"""
    
    # Non-degenerate ellipsoid
    E1 = Ellipsoid(np.eye(2), np.array([[0], [0]]))
    
    # Point ellipsoid (all-zero shape matrix)
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [1]]))
    
    # Distance should equal distance to the center point
    dist_ellipsoid = E1.distance(E_point)
    dist_point = E1.distance(E_point.q)
    
    assert np.abs(dist_ellipsoid - dist_point) < 1e-10, \
        f"Distance to point ellipsoid should equal distance to its center"


def test_distance_error_cases():
    """Test error handling in distance computation"""
    
    # Different dimensions should raise error
    E1 = Ellipsoid(np.eye(2))
    E2 = Ellipsoid(np.eye(3))
    
    with pytest.raises(ValueError, match="Dimensions"):
        E1.distance(E2)


def test_distance_multiple_sets():
    """Test distance to multiple sets (list input)"""
    
    E = Ellipsoid(np.eye(2))
    
    # List of points
    points = [
        np.array([[0.5], [0]]),    # inside  
        np.array([[1.5], [0]]),    # outside
        np.array([[1.0], [0]])     # on boundary
    ]
    
    distances = E.distance(points)
    
    assert len(distances) == 3, "Should return distance for each point"
    assert distances[0] < 0, "First point should be inside"
    assert distances[1] > 0, "Second point should be outside" 
    assert np.abs(distances[2]) < 1e-10, "Third point should be on boundary"


def test_distance_empty_set():
    """Test distance to empty set"""
    
    E = Ellipsoid(np.eye(2))
    
    try:
        from cora_python.contSet.emptySet import EmptySet
        empty_set = EmptySet(2)
        
        # Distance to empty set should be 0
        dist = E.distance(empty_set)
        assert dist == 0.0, f"Distance to empty set should be 0, got {dist}"
        
    except ImportError:
        pytest.skip("EmptySet class not available")


# Additional integration tests

def test_distance_with_tolerance():
    """Test that distance respects ellipsoid tolerance settings"""
    
    # Ellipsoid with specific tolerance
    tol = 1e-8
    E = Ellipsoid(np.eye(2), tol=tol)
    
    # Point very close to boundary
    p = np.array([[1.0 + tol/2], [0]])
    dist = E.distance(p)
    
    # Distance should be small but positive
    assert 0 < dist < tol, f"Distance should respect tolerance, got {dist}"


@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_distance_various_dimensions(dim):
    """Test distance computation in various dimensions"""
    
    # Create ellipsoid and test point
    E = Ellipsoid(np.eye(dim))
    p_out = np.ones((dim, 1)) * 2  # Point outside in all dimensions
    
    dist = E.distance(p_out)
    assert dist > 0, f"Distance should be positive in {dim}D, got {dist}"


if __name__ == "__main__":
    test_ellipsoid_distance()
    test_distance_point_basic()
    test_distance_ellipsoid_to_ellipsoid()
    test_distance_degenerate_ellipsoid()
    test_distance_error_cases()
    test_distance_multiple_sets()
    test_distance_empty_set()
    test_distance_with_tolerance()
    print("All distance tests passed!") 