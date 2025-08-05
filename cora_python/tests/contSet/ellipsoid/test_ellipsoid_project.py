"""
test_ellipsoid_project - unit test function of project

This module tests the ellipsoid project implementation exactly matching MATLAB.

Authors:       Mark Wetzlinger (MATLAB), Python translation by AI Assistant
Written:       27-August-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_ellipsoid_project():
    """Main project test matching MATLAB test_ellipsoid_project"""
    
    # Create a random PSD Q matrix (using fixed seed for reproducibility)
    np.random.seed(42)
    n = 4
    O = np.linalg.qr(np.random.randn(n, n))[0]  # Orthogonal matrix
    D = np.diag(np.abs(np.random.randn(n)) + 0.3)  # Positive eigenvalues
    Q = O @ D @ O.T
    
    # Generate ellipsoid
    E = Ellipsoid(Q)
    
    # Project ellipsoid using range [1, 2] (0-based indexing)
    projDim = [1, 2]  # This means indices 1, 2 in 0-based Python (dimensions 2, 3)
    E_proj1 = E.project(projDim)
    
    # True solution (extract submatrix)
    # Python: Q[1:3, 1:3] for 0-based indexing (range [1, 2] inclusive)
    start_idx = projDim[0]
    end_idx = projDim[1] + 1  # Make end inclusive
    Q_proj = Q[start_idx:end_idx, start_idx:end_idx]
    E_true = Ellipsoid(Q_proj)
    
    # Logical indexing
    projDim_logical = [False, True, True, False]  # Project to dimensions 1, 2 (0-based)
    E_proj2 = E.project(projDim_logical)
    
    # Check properties
    assert np.all(withinTol(E_true.Q, E_proj1.Q, E.TOL)), \
        f"Projected Q matrix mismatch: expected {E_true.Q}, got {E_proj1.Q}"
    assert E_true.dim() == E_proj1.dim(), \
        f"Projected dimension mismatch: expected {E_true.dim()}, got {E_proj1.dim()}"
    assert np.all(withinTol(E_true.Q, E_proj2.Q, E.TOL)), \
        f"Logical projection Q matrix mismatch: expected {E_true.Q}, got {E_proj2.Q}"
    assert E_true.dim() == E_proj2.dim(), \
        f"Logical projection dimension mismatch: expected {E_true.dim()}, got {E_proj2.dim()}"


def test_project_with_center():
    """Test projection of ellipsoid with non-zero center"""
    
    Q = np.array([[4, 1, 0], [1, 2, 0.5], [0, 0.5, 1]])
    q = np.array([[1], [2], [3]])
    E = Ellipsoid(Q, q)
    
    # Project to first two dimensions
    E_proj = E.project([0, 1])  # 0-based indexing for first two dimensions
    
    # Expected: Q[0:2, 0:2] and q[0:2]
    expected_Q = Q[0:2, 0:2]
    expected_q = q[0:2, :]
    
    assert np.allclose(E_proj.Q, expected_Q), \
        f"Projected Q with center incorrect: got {E_proj.Q}, expected {expected_Q}"
    assert np.allclose(E_proj.q, expected_q), \
        f"Projected center incorrect: got {E_proj.q}, expected {expected_q}"


def test_project_logical_indexing():
    """Test projection using logical indexing"""
    
    Q = np.eye(4)
    q = np.array([[1], [2], [3], [4]])
    E = Ellipsoid(Q, q)
    
    # Logical projection: select dimensions 0, 2 (Python 0-based)
    logical_dims = [True, False, True, False]
    E_proj = E.project(logical_dims)
    
    # Expected result
    selected_indices = [0, 2]
    expected_Q = Q[np.ix_(selected_indices, selected_indices)]
    expected_q = q[selected_indices, :]
    
    assert np.allclose(E_proj.Q, expected_Q), \
        f"Logical projection Q incorrect: got {E_proj.Q}, expected {expected_Q}"
    assert np.allclose(E_proj.q, expected_q), \
        f"Logical projection center incorrect: got {E_proj.q}, expected {expected_q}"


def test_project_single_dimension():
    """Test projection to single dimension"""
    
    Q = np.array([[2, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 1]])
    q = np.array([[5], [-2], [1]])
    E = Ellipsoid(Q, q)
    
    # Project to second dimension only (0-based index 1)
    E_proj = E.project([1])
    
    # Expected 1D result
    expected_Q = np.array([[Q[1, 1]]])
    expected_q = np.array([[q[1, 0]]])
    
    assert E_proj.dim() == 1, f"Single dimension projection should be 1D, got {E_proj.dim()}"
    assert np.allclose(E_proj.Q, expected_Q), \
        f"Single dimension Q incorrect: got {E_proj.Q}, expected {expected_Q}"
    assert np.allclose(E_proj.q, expected_q), \
        f"Single dimension center incorrect: got {E_proj.q}, expected {expected_q}"


def test_project_consecutive_dimensions():
    """Test projection to consecutive dimensions"""
    
    Q = np.random.rand(5, 5)
    Q = Q @ Q.T  # Make PSD
    q = np.random.randn(5, 1)
    E = Ellipsoid(Q, q)
    
    # Project to dimensions 1-3 (0-based indexing, range syntax)
    E_proj = E.project([1, 3])
    
    # Expected: dimensions 1-3 in 0-based indexing (inclusive)
    expected_Q = Q[1:4, 1:4]
    expected_q = q[1:4, :]
    
    assert E_proj.dim() == 3, f"Consecutive projection should be 3D, got {E_proj.dim()}"
    assert np.allclose(E_proj.Q, expected_Q), \
        f"Consecutive projection Q incorrect"
    assert np.allclose(E_proj.q, expected_q), \
        f"Consecutive projection center incorrect"


def test_project_empty_ellipsoid():
    """Test projection of empty ellipsoid"""
    
    E_empty = Ellipsoid.empty(4)
    E_proj = E_empty.project([0, 1])
    
    # Projection of empty ellipsoid should remain empty
    assert E_proj.representsa_('emptySet', E_proj.TOL), \
        "Projection of empty ellipsoid should remain empty"
    assert E_proj.dim() == 2, f"Projected empty ellipsoid should be 2D, got {E_proj.dim()}"


def test_project_degenerate_ellipsoid():
    """Test projection of degenerate ellipsoid"""
    
    # Create degenerate ellipsoid (rank deficient)
    Q = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    q = np.array([[1], [2], [3]])
    E = Ellipsoid(Q, q)
    
    # Project to all dimensions
    E_proj = E.project([0, 1, 2])
    
    # Should be the same as original
    assert np.allclose(E_proj.Q, E.Q), "Full projection should be identity"
    assert np.allclose(E_proj.q, E.q), "Full projection center should be unchanged"
    
    # Project to degenerate dimension
    E_proj_deg = E.project([2])  # Middle dimension (index 1)
    expected_Q = np.array([[0]])
    expected_q = np.array([[2]])
    
    assert np.allclose(E_proj_deg.Q, expected_Q), \
        f"Degenerate dimension projection incorrect: got {E_proj_deg.Q}, expected {expected_Q}"
    assert np.allclose(E_proj_deg.q, expected_q), \
        f"Degenerate dimension center incorrect: got {E_proj_deg.q}, expected {expected_q}"


def test_project_error_cases():
    """Test error handling in project method"""
    
    E = Ellipsoid(np.eye(3))
    
    # Invalid dimension index (too large)
    with pytest.raises(ValueError):
        E.project([0, 5])  # Index 5 doesn't exist in 3D ellipsoid
    
    # Invalid dimension index (negative)
    with pytest.raises(ValueError):
        E.project([-1])  # Negative index should be invalid
    
    # Empty projection dimensions
    with pytest.raises(ValueError):
        E.project([])
    
    # Invalid logical indexing (wrong length)
    with pytest.raises(ValueError):
        E.project([True, False])  # Should have 3 elements for 3D ellipsoid


def test_project_preserves_properties():
    """Test that projection preserves ellipsoid properties"""
    
    Q = np.array([[2, 1], [1, 3]])
    q = np.array([[1], [2]])
    E = Ellipsoid(Q, q, tol=1e-10)
    
    # Project to single dimension
    E_proj = E.project([1])
    
    # Tolerance should be preserved
    assert E_proj.TOL == E.TOL, f"Tolerance should be preserved: expected {E.TOL}, got {E_proj.TOL}"
    
    # Projected matrix should still be PSD
    assert np.all(np.linalg.eigvals(E_proj.Q) >= -E_proj.TOL), \
        "Projected shape matrix should remain positive semidefinite"


@pytest.mark.parametrize("start_dim,end_dim", [(1, 2), (1, 3), (2, 3), (2, 4)])
def test_project_various_ranges(start_dim, end_dim):
    """Test projection with various dimension ranges"""
    
    Q = np.eye(4) + 0.1 * np.random.randn(4, 4)
    Q = Q @ Q.T  # Make PSD
    E = Ellipsoid(Q)
    
    E_proj = E.project([start_dim, end_dim])
    
    expected_dim = end_dim - start_dim + 1
    assert E_proj.dim() == expected_dim, \
        f"Projected dimension should be {expected_dim}, got {E_proj.dim()}"


if __name__ == "__main__":
    test_ellipsoid_project()
    test_project_with_center()
    test_project_logical_indexing()
    test_project_single_dimension()
    test_project_consecutive_dimensions()
    test_project_empty_ellipsoid()
    test_project_degenerate_ellipsoid()
    test_project_error_cases()
    test_project_preserves_properties()
    print("All project tests passed!") 