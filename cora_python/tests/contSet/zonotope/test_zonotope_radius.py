"""
Test radius method for zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope


def test_radius_simple():
    """Test radius for simple zonotope"""
    # Simple 2D zonotope with unit generators
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: sum of generator norms = norm([1,0]) + norm([0,1]) = 1 + 1 = 2
    # Method 2: interval radius = norm of half-edge lengths
    # The interval is [-1,1] x [-1,1], so radius = norm([1,1]) = sqrt(2)
    # Minimum should be sqrt(2) ≈ 1.414
    expected = np.sqrt(2)
    assert np.isclose(r, expected, atol=1e-10), f"Expected {expected}, got {r}"


def test_radius_1d():
    """Test radius for 1D zonotope"""
    # 1D zonotope: [1, 3] (center=2, generator=1)
    c = np.array([2])
    G = np.array([[1]])
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: sum of generator norms = 1
    # Method 2: interval radius = 1 (half of edge length 2)
    # Both methods give 1
    assert np.isclose(r, 1), f"Expected 1, got {r}"


def test_radius_multiple_generators():
    """Test radius with multiple generators"""
    c = np.array([0, 0])
    G = np.array([[1, 2], [0, 1]])  # Two generators: [1,0] and [2,1]
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: norm([1,0]) + norm([2,1]) = 1 + sqrt(5) ≈ 3.236
    # Method 2: interval is [-3,3] x [-1,1], radius = norm([3,1]) = sqrt(10) ≈ 3.162
    # Minimum should be sqrt(10)
    expected = np.sqrt(10)
    assert np.isclose(r, expected, atol=1e-10), f"Expected {expected}, got {r}"


def test_radius_zero_generators():
    """Test radius with no generators (point)"""
    c = np.array([1, 2])
    G = np.array([[], []])  # No generators
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # A point has radius 0
    assert np.isclose(r, 0), f"Expected 0, got {r}"


def test_radius_single_generator():
    """Test radius with single generator"""
    c = np.array([0, 0])
    G = np.array([[3], [4]])  # Single generator [3, 4]
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: norm([3,4]) = 5
    # Method 2: interval is [-3,3] x [-4,4], radius = norm([3,4]) = 5
    # Both methods give 5
    assert np.isclose(r, 5), f"Expected 5, got {r}"


def test_radius_translated_zonotope():
    """Test radius for translated zonotope (center affects interval but not generator sum)"""
    c = np.array([10, 20])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: sum of generator norms = 2 (independent of center)
    # Method 2: interval is [9,11] x [19,21], radius = norm([1,1]) = sqrt(2)
    # Minimum should be sqrt(2)
    expected = np.sqrt(2)
    assert np.isclose(r, expected, atol=1e-10), f"Expected {expected}, got {r}"


def test_radius_different_scales():
    """Test radius with generators of different scales"""
    c = np.array([0, 0])
    G = np.array([[0.1, 10], [0.1, 0]])  # Small and large generators
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: norm([0.1, 0.1]) + norm([10, 0]) = sqrt(0.02) + 10 ≈ 10.141
    # Method 2: interval is [-10.1, 10.1] x [-0.1, 0.1], radius = norm([10.1, 0.1]) ≈ 10.1005
    # Minimum should be the interval method result
    method1 = np.sqrt(0.02) + 10
    method2 = np.sqrt(10.1**2 + 0.1**2)
    expected = min(method1, method2)
    assert np.isclose(r, expected, atol=1e-10), f"Expected {expected}, got {r}"


def test_radius_3d():
    """Test radius for 3D zonotope"""
    c = np.array([0, 0, 0])
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Unit cube
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: 1 + 1 + 1 = 3
    # Method 2: interval is [-1,1]³, radius = norm([1,1,1]) = sqrt(3)
    # Minimum should be sqrt(3)
    expected = np.sqrt(3)
    assert np.isclose(r, expected, atol=1e-10), f"Expected {expected}, got {r}"


def test_radius_negative_generators():
    """Test radius with negative generator components"""
    c = np.array([0, 0])
    G = np.array([[-1, 2], [3, -1]])
    Z = Zonotope(c, G)
    
    r = Z.radius()
    
    # Method 1: norm([-1,3]) + norm([2,-1]) = sqrt(10) + sqrt(5)
    # Method 2: interval is [-3,3] x [-1,4], radius = norm([3,4]) = 5
    method1 = np.sqrt(10) + np.sqrt(5)
    method2 = 5
    expected = min(method1, method2)
    assert np.isclose(r, expected, atol=1e-10), f"Expected {expected}, got {r}"


def test_radius_empty_zonotope():
    """Test radius for empty zonotope"""
    Z = Zonotope.empty(2)
    
    r = Z.radius()
    
    # Empty zonotope should have radius 0
    assert np.isclose(r, 0), f"Expected 0, got {r}"


def test_radius_consistency():
    """Test that radius is always non-negative and consistent"""
    test_cases = [
        (np.array([0]), np.array([[1]])),
        (np.array([0, 0]), np.array([[1, 0], [0, 1]])),
        (np.array([1, 2]), np.array([[2, -1], [1, 3]])),
        (np.array([0, 0, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
    ]
    
    for c, G in test_cases:
        Z = Zonotope(c, G)
        r = Z.radius()
        
        # Radius should be non-negative
        assert r >= 0, f"Radius should be non-negative, got {r}"
        
        # Radius should be finite
        assert np.isfinite(r), f"Radius should be finite, got {r}"


if __name__ == "__main__":
    test_radius_simple()
    test_radius_1d()
    test_radius_multiple_generators()
    test_radius_zero_generators()
    test_radius_single_generator()
    test_radius_translated_zonotope()
    test_radius_different_scales()
    test_radius_3d()
    test_radius_negative_generators()
    test_radius_empty_zonotope()
    test_radius_consistency()
    print("All radius tests passed!") 