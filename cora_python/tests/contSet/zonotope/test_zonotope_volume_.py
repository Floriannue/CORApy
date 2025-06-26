"""
Test volume_ method for zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_volume_exact_2d():
    """Test volume_ exact computation for 2D zonotope"""
    # Unit square: center=[0,0], generators=[[1,0],[0,1]]
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Unit square has volume 4 (extends from -1 to 1 in both dimensions)
    expected = 4.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_exact_1d():
    """Test volume_ exact computation for 1D zonotope"""
    # 1D interval [1, 3]: center=2, generator=1
    c = np.array([2])
    G = np.array([[1]])
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # 1D "volume" is length = 2
    expected = 2.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_exact_parallelotope():
    """Test volume_ for parallelotope (2 generators in 2D)"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0.5, 1]])  # Parallelogram
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Volume = 2^n * |det(G)| = 4 * |det([[1,0],[0.5,1]])| = 4 * 1 = 4
    expected = 4.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_exact_3d():
    """Test volume_ for 3D zonotope"""
    c = np.array([0, 0, 0])
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Unit cube
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Unit cube has volume 2^3 = 8
    expected = 8.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_exact_redundant_generators():
    """Test volume_ with more generators than dimensions"""
    c = np.array([0, 0])
    G = np.array([[1, 0, 0.5], [0, 1, 0.5]])  # 3 generators in 2D
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Should compute volume considering all combinations
    # C(3,2) = 3 combinations: (0,1), (0,2), (1,2)
    # det([[1,0],[0,1]]) = 1, det([[1,0.5],[0,0.5]]) = 0.5, det([[0,0.5],[1,0.5]]) = -0.5
    # Volume = 2^2 * (1 + 0.5 + 0.5) = 4 * 2 = 8
    expected = 8.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_degenerate():
    """Test volume_ for degenerate zonotope (rank deficient)"""
    c = np.array([0, 0])
    G = np.array([[1, 2], [2, 4]])  # Linearly dependent generators
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Degenerate zonotope has volume 0
    assert np.isclose(vol, 0), f"Expected 0, got {vol}"


def test_volume_insufficient_generators():
    """Test volume_ with insufficient generators"""
    c = np.array([0, 0])
    G = np.array([[1], [2]])  # Only 1 generator in 2D
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Insufficient generators for full dimension -> volume 0
    assert np.isclose(vol, 0), f"Expected 0, got {vol}"


def test_volume_alamo():
    """Test volume_ with alamo approximation"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    vol = Z.volume_('alamo')
    
    # Alamo: 2^n * sqrt(det(G*G')) = 4 * sqrt(det([[1,0],[0,1]])) = 4 * 1 = 4
    expected = 4.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_alamo_non_orthogonal():
    """Test volume_ alamo with non-orthogonal generators"""
    c = np.array([0, 0])
    G = np.array([[1, 1], [0, 1]])
    Z = Zonotope(c, G)
    
    vol = Z.volume_('alamo')
    
    # G*G' = [[2, 1], [1, 1]], det = 2-1 = 1, volume = 4 * sqrt(1) = 4
    expected = 4.0
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


def test_volume_reduce():
    """Test volume_ with reduce method"""
    c = np.array([0, 0])
    G = np.array([[1, 0, 0.1], [0, 1, 0.1]])  # 3 generators
    Z = Zonotope(c, G)
    
    vol = Z.volume_('reduce', order=1)
    
    # Should reduce to order 1 (2 generators for 2D) and compute volume
    assert vol > 0, f"Volume should be positive, got {vol}"


def test_volume_reduce_no_order():
    """Test volume_ reduce method without order parameter"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    with pytest.raises(CORAerror):
        Z.volume_('reduce')


def test_volume_invalid_method():
    """Test volume_ with invalid method"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    with pytest.raises(CORAerror):
        Z.volume_('invalid')


def test_volume_empty_zonotope():
    """Test volume_ for empty zonotope"""
    Z = Zonotope.empty(2)
    
    vol = Z.volume_('exact')
    
    # Empty zonotope has volume 0
    assert np.isclose(vol, 0), f"Expected 0, got {vol}"


def test_volume_zero_generators():
    """Test volume_ with no generators (point)"""
    c = np.array([1, 2])
    G = np.array([[], []])  # No generators
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # Point has volume 0
    assert np.isclose(vol, 0), f"Expected 0, got {vol}"


def test_volume_scaling():
    """Test volume_ with scaled generators"""
    c = np.array([0, 0])
    G1 = np.array([[1, 0], [0, 1]])
    G2 = np.array([[2, 0], [0, 2]])  # Scaled by factor 2
    Z1 = Zonotope(c, G1)
    Z2 = Zonotope(c, G2)
    
    vol1 = Z1.volume_('exact')
    vol2 = Z2.volume_('exact')
    
    # Volume should scale by factor^dimension = 2^2 = 4
    assert np.isclose(vol2, 4 * vol1), f"Expected {4 * vol1}, got {vol2}"


def test_volume_consistency():
    """Test that different methods give consistent results for simple cases"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    vol_exact = Z.volume_('exact')
    vol_alamo = Z.volume_('alamo')
    
    # For orthogonal generators, exact and alamo should be the same
    assert np.isclose(vol_exact, vol_alamo), f"Exact: {vol_exact}, Alamo: {vol_alamo}"


def test_volume_large_dimension():
    """Test volume_ for higher dimensional zonotope"""
    n = 4
    c = np.zeros(n)
    G = np.eye(n)  # Identity matrix
    Z = Zonotope(c, G)
    
    vol = Z.volume_('exact')
    
    # n-dimensional unit hypercube has volume 2^n
    expected = 2**n
    assert np.isclose(vol, expected), f"Expected {expected}, got {vol}"


if __name__ == "__main__":
    test_volume_exact_2d()
    test_volume_exact_1d()
    test_volume_exact_parallelotope()
    test_volume_exact_3d()
    test_volume_exact_redundant_generators()
    test_volume_degenerate()
    test_volume_insufficient_generators()
    test_volume_alamo()
    test_volume_alamo_non_orthogonal()
    test_volume_reduce()
    test_volume_reduce_no_order()
    test_volume_invalid_method()
    test_volume_empty_zonotope()
    test_volume_zero_generators()
    test_volume_scaling()
    test_volume_consistency()
    test_volume_large_dimension()
    print("All volume_ tests passed!") 