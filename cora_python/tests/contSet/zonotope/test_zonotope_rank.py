"""
Test rank method for zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope


def test_rank_full_rank():
    """Test rank for full rank zonotope"""
    # 2D zonotope with 2 linearly independent generators
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 2, f"Expected rank 2, got {r}"


def test_rank_reduced_rank():
    """Test rank for reduced rank zonotope"""
    # 2D zonotope with linearly dependent generators
    c = np.array([0, 0])
    G = np.array([[1, 2], [2, 4]])  # Second column is 2 * first column
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 1, f"Expected rank 1, got {r}"


def test_rank_1d():
    """Test rank for 1D zonotope"""
    c = np.array([2])
    G = np.array([[1, 3]])  # Two generators in 1D
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 1, f"Expected rank 1, got {r}"


def test_rank_zero_generators():
    """Test rank with no generators (point)"""
    c = np.array([1, 2])
    G = np.array([[], []])  # No generators
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 0, f"Expected rank 0, got {r}"


def test_rank_single_generator():
    """Test rank with single generator"""
    c = np.array([0, 0, 0])
    G = np.array([[1], [2], [3]])  # Single generator in 3D
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 1, f"Expected rank 1, got {r}"


def test_rank_zero_generator():
    """Test rank with zero generator"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [2, 0]])  # One real generator, one zero
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 1, f"Expected rank 1, got {r}"


def test_rank_multiple_zero_generators():
    """Test rank with multiple zero generators"""
    c = np.array([0, 0])
    G = np.array([[0, 0, 1], [0, 0, 2]])  # Two zero generators, one real
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 1, f"Expected rank 1, got {r}"


def test_rank_3d_full():
    """Test rank for full rank 3D zonotope"""
    c = np.array([0, 0, 0])
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 3, f"Expected rank 3, got {r}"


def test_rank_3d_planar():
    """Test rank for planar zonotope in 3D"""
    c = np.array([0, 0, 0])
    G = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])  # All generators in xy-plane
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 2, f"Expected rank 2, got {r}"


def test_rank_with_tolerance():
    """Test rank with tolerance parameter"""
    c = np.array([0, 0])
    G = np.array([[1, 1e-10], [0, 1e-10]])  # Second generator is nearly zero
    Z = Zonotope(c, G)
    
    # Without tolerance, might be rank 2 due to numerical precision
    r_no_tol = Z.rank()
    
    # With tolerance, should be rank 1
    r_with_tol = Z.rank(tol=1e-8)
    assert r_with_tol == 1, f"Expected rank 1 with tolerance, got {r_with_tol}"


def test_rank_linearly_dependent_3_vectors():
    """Test rank with 3 linearly dependent vectors in 2D"""
    c = np.array([0, 0])
    G = np.array([[1, 2, 3], [2, 4, 6]])  # All columns are multiples of [1, 2]
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 1, f"Expected rank 1, got {r}"


def test_rank_mixed_dependencies():
    """Test rank with mixed linear dependencies"""
    c = np.array([0, 0, 0])
    G = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [1, 1, 2, 2]])  # Third column = first + second, fourth = 2*first
    Z = Zonotope(c, G)
    
    r = Z.rank()
    assert r == 2, f"Expected rank 2, got {r}"


def test_rank_empty_zonotope():
    """Test rank for empty zonotope"""
    Z = Zonotope.empty(3)
    
    r = Z.rank()
    assert r == 0, f"Expected rank 0 for empty zonotope, got {r}"


def test_rank_consistency():
    """Test that rank is consistent and bounded"""
    test_cases = [
        (np.array([0]), np.array([[1, 2, 3]])),
        (np.array([0, 0]), np.array([[1, 0], [0, 1]])),
        (np.array([0, 0]), np.array([[1, 2], [2, 4]])),
        (np.array([0, 0, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
    ]
    
    for c, G in test_cases:
        Z = Zonotope(c, G)
        r = Z.rank()
        
        # Rank should be non-negative
        assert r >= 0, f"Rank should be non-negative, got {r}"
        
        # Rank should not exceed the dimension
        assert r <= len(c), f"Rank {r} should not exceed dimension {len(c)}"
        
        # Rank should not exceed number of generators
        assert r <= G.shape[1], f"Rank {r} should not exceed number of generators {G.shape[1]}"


def test_rank_comparison_with_numpy():
    """Test that our rank matches numpy's matrix_rank"""
    test_matrices = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 2], [2, 4]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[0, 0], [0, 0]]),
    ]
    
    for G in test_matrices:
        c = np.zeros(G.shape[0])
        Z = Zonotope(c, G)
        
        expected_rank = np.linalg.matrix_rank(G)
        actual_rank = Z.rank()
        
        assert actual_rank == expected_rank, f"Expected rank {expected_rank}, got {actual_rank} for matrix {G}"


if __name__ == "__main__":
    test_rank_full_rank()
    test_rank_reduced_rank()
    test_rank_1d()
    test_rank_zero_generators()
    test_rank_single_generator()
    test_rank_zero_generator()
    test_rank_multiple_zero_generators()
    test_rank_3d_full()
    test_rank_3d_planar()
    test_rank_with_tolerance()
    test_rank_linearly_dependent_3_vectors()
    test_rank_mixed_dependencies()
    test_rank_empty_zonotope()
    test_rank_consistency()
    test_rank_comparison_with_numpy()
    print("All rank tests passed!") 