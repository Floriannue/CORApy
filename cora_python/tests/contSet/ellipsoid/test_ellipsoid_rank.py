"""
test_ellipsoid_rank - unit test function of rank

This module tests the ellipsoid rank implementation exactly matching MATLAB.

Authors:       Victor Gassmann (MATLAB), Python translation by AI Assistant
Written:       27-July-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


def test_ellipsoid_rank():
    """Main rank test matching MATLAB test_ellipsoid_rank"""
    
    # Empty case: rank = 0
    E_empty = Ellipsoid.empty(2)
    assert E_empty.rank() == 0, f"Empty ellipsoid rank should be 0, got {E_empty.rank()}"
    
    # Init cases (exact MATLAB values)
    E1 = Ellipsoid(
        np.array([[5.4387811500952807, 12.4977183618314545], 
                  [12.4977183618314545, 29.6662117284481646]]), 
        np.array([[-0.7445068341257537], [3.5800647524843665]]),
        0.000001
    )
    Ed1 = Ellipsoid(
        np.array([[4.2533342807136076, 0.6346400221575308], 
                  [0.6346400221575309, 0.0946946398147988]]), 
        np.array([[-2.4653656883489115], [0.2717868749873985]]),
        0.000001
    )
    E0 = Ellipsoid(
        np.array([[0.0000000000000000, 0.0000000000000000], 
                  [0.0000000000000000, 0.0000000000000000]]), 
        np.array([[1.0986933635979599], [-1.9884387759871638]]),
        0.000001
    )
    n = E1.dim()
    
    # Test rank values
    assert E1.rank() == n, f"E1 rank should be {n}, got {E1.rank()}"
    assert Ed1.rank() != n, f"Ed1 rank should not be {n}, got {Ed1.rank()}"
    assert Ed1.rank() > E0.rank(), f"Ed1 rank ({Ed1.rank()}) should be greater than E0 rank ({E0.rank()})"
    assert E0.rank() == 0, f"E0 rank should be 0, got {E0.rank()}"


def test_rank_various_cases():
    """Test rank computation for various ellipsoid types"""
    
    # Full rank ellipsoid
    E_full = Ellipsoid(np.eye(3))
    assert E_full.rank() == 3, "Full rank 3D ellipsoid should have rank 3"
    
    # Rank deficient ellipsoid (one zero eigenvalue)
    Q_deficient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E_deficient = Ellipsoid(Q_deficient)
    assert E_deficient.rank() == 2, f"Rank deficient ellipsoid should have rank 2, got {E_deficient.rank()}"
    
    # Point ellipsoid (zero rank)
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [1]]))
    assert E_point.rank() == 0, f"Point ellipsoid should have rank 0, got {E_point.rank()}"


def test_rank_with_tolerance():
    """Test rank computation with different tolerances"""
    
    # Create ellipsoid with small eigenvalue
    small_val = 1e-10
    Q = np.diag([1, small_val])
    E = Ellipsoid(Q, tol=1e-8)
    
    # With default tolerance, small eigenvalue should be considered zero
    rank_tight = E.rank()
    
    # With looser tolerance, small eigenvalue might be kept
    E_loose = Ellipsoid(Q, tol=1e-12)
    rank_loose = E_loose.rank()
    
    # The rank with tighter tolerance should be less than or equal to loose tolerance
    assert rank_tight <= rank_loose, f"Rank with tighter tolerance ({rank_tight}) should be <= loose tolerance ({rank_loose})"


def test_rank_edge_cases():
    """Test rank computation for edge cases"""
    
    # Very small ellipsoid
    E_small = Ellipsoid(1e-15 * np.eye(2))
    rank_small = E_small.rank()
    assert 0 <= rank_small <= 2, f"Small ellipsoid rank should be between 0 and 2, got {rank_small}"
    
    # Large ellipsoid
    E_large = Ellipsoid(1e6 * np.eye(2))
    assert E_large.rank() == 2, f"Large ellipsoid should have full rank, got {E_large.rank()}"


@pytest.mark.parametrize("dim", [1, 2, 3, 5])
def test_rank_various_dimensions(dim):
    """Test rank computation in various dimensions"""
    
    # Full rank ellipsoid
    E_full = Ellipsoid(np.eye(dim))
    assert E_full.rank() == dim, f"Full rank {dim}D ellipsoid should have rank {dim}"
    
    # Rank deficient (if possible)
    if dim > 1:
        Q_deficient = np.eye(dim)
        Q_deficient[-1, -1] = 0  # Make last eigenvalue zero
        E_deficient = Ellipsoid(Q_deficient)
        assert E_deficient.rank() == dim - 1, f"Rank deficient {dim}D ellipsoid should have rank {dim-1}"


def test_rank_consistency_with_isFullDim():
    """Test that rank is consistent with isFullDim"""
    
    # Full rank ellipsoid
    E_full = Ellipsoid(np.eye(2))
    assert E_full.rank() == E_full.dim() == 2
    assert E_full.isFullDim(), "Full rank ellipsoid should be full dimensional"
    
    # Rank deficient ellipsoid
    Q_deficient = np.array([[1, 0], [0, 0]])
    E_deficient = Ellipsoid(Q_deficient)
    assert E_deficient.rank() < E_deficient.dim()
    assert not E_deficient.isFullDim(), "Rank deficient ellipsoid should not be full dimensional"


if __name__ == "__main__":
    test_ellipsoid_rank()
    test_rank_various_cases()
    test_rank_with_tolerance()
    test_rank_edge_cases()
    test_rank_consistency_with_isFullDim()
    print("All rank tests passed!") 