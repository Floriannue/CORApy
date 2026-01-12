"""
GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/global/functions/helper/dynamics/contDynamics/linearSys/arnoldi.m
Generated: 2025-01-XX

Tests the Arnoldi iteration for building a Krylov subspace.
"""

import numpy as np
import pytest
from cora_python.g.functions.helper.dynamics.contDynamics.linearSys.arnoldi import arnoldi


def test_arnoldi_01_basic():
    """
    GENERATED TEST - Basic functionality test
    
    Tests the core functionality of arnoldi based on MATLAB source code.
    Verifies that:
    1. Arnoldi iteration produces orthonormal basis V
    2. H is upper Hessenberg matrix
    3. Happy breakdown detection works
    4. V and H satisfy the Arnoldi relation: A*V = V*H + H(j+1,j)*v_{j+1}*e_j^T
    """
    # Setup: Create a simple test matrix
    # Use a 5x5 matrix for testing
    np.random.seed(42)  # For reproducibility
    A = np.random.randn(5, 5)
    A = (A + A.T) / 2  # Make symmetric for easier testing
    
    # Initial vector
    vInit = np.random.randn(5, 1)
    vInit = vInit / np.linalg.norm(vInit)  # Normalize
    
    # Reduced dimension
    redDim = 3
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    # 1. V should have correct dimensions
    assert V.shape == (5, redDim), f"V should have shape (5, {redDim}), got {V.shape}"
    
    # 2. V should be orthonormal (columns are orthonormal)
    VTV = V.T @ V
    np.testing.assert_allclose(VTV, np.eye(redDim), atol=1e-10,
                               err_msg="V should be orthonormal")
    
    # 3. H should be upper Hessenberg (zeros below first subdiagonal)
    for i in range(H.shape[0]):
        for j in range(i - 1):  # j < i-1 should be zero
            assert abs(H[i, j]) < 1e-10, f"H[{i},{j}] should be zero (upper Hessenberg)"
    
    # 4. Hlast should be the last computed H(j+1,j) value
    # Note: Hlast is the value from H[redDim, redDim-1] before the last row was removed
    # This is the last subdiagonal element that was computed but then removed
    if not happyBreakdown:
        # Hlast should be a positive value (norm of a vector)
        assert Hlast > 0, "Hlast should be positive"
        # For verification, we can check that Hlast is reasonable (not zero, not NaN)
        assert np.isfinite(Hlast), "Hlast should be finite"
    
    # 5. Arnoldi relation: A*V(:,1:j) = V*H + H(j+1,j)*V(:,j+1)*e_j^T
    # For the last column (if no happy breakdown), check: A*V = V*H + Hlast*V_next*e_j^T
    # Simplified check: V.T @ A @ V should be approximately H (for symmetric A)
    if not happyBreakdown:
        VTAV = V.T @ A @ V
        # H should be approximately V.T @ A @ V (for symmetric A)
        np.testing.assert_allclose(H, VTAV, atol=1e-6,
                                   err_msg="Arnoldi relation should hold")


def test_arnoldi_02_happy_breakdown():
    """
    GENERATED TEST - Happy breakdown test
    
    Tests that happy breakdown is detected when the Krylov subspace
    dimension is smaller than requested.
    """
    # Setup: Create a matrix that will cause early termination
    # Use a low-rank matrix
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])  # Rank 2 matrix
    
    vInit = np.array([[1], [0], [0]])
    redDim = 5  # Request more dimensions than possible
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    # Should detect happy breakdown
    # The actual dimension will be less than redDim
    assert happyBreakdown or V.shape[1] < redDim, "Should detect happy breakdown or reduce dimension"
    
    # V should still be orthonormal
    if V.shape[1] > 0:
        VTV = V.T @ V
        np.testing.assert_allclose(VTV, np.eye(V.shape[1]), atol=1e-10,
                                   err_msg="V should be orthonormal even with happy breakdown")


def test_arnoldi_03_zero_vector_error():
    """
    GENERATED TEST - Error handling test
    
    Tests that arnoldi raises an error for zero initial vector.
    """
    A = np.random.randn(3, 3)
    vInit = np.zeros((3, 1))
    redDim = 2
    
    # Execute and verify
    with pytest.raises(ValueError, match="vInit cannot be zero"):
        arnoldi(A, vInit, redDim)


def test_arnoldi_04_small_system():
    """
    GENERATED TEST - Small system test
    
    Tests arnoldi on a 2x2 system.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    vInit = np.array([[1], [0]])
    redDim = 2
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    assert V.shape == (2, 2), "V should have shape (2, 2)"
    assert H.shape == (2, 2), "H should have shape (2, 2)"
    assert not happyBreakdown, "Should not have happy breakdown for full dimension"
    
    # Check orthonormality
    VTV = V.T @ V
    np.testing.assert_allclose(VTV, np.eye(2), atol=1e-10)


def test_arnoldi_05_sparse_matrix():
    """
    GENERATED TEST - Sparse matrix test
    
    Tests that arnoldi handles sparse matrices correctly.
    """
    from scipy.sparse import csc_matrix
    
    # Create sparse matrix
    A = csc_matrix([[1, 2, 0],
                    [0, 3, 4],
                    [5, 0, 6]])
    
    vInit = np.array([[1], [0], [0]])
    redDim = 2
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    assert V.shape == (3, 2), "V should have shape (3, 2)"
    assert H.shape == (2, 2), "H should have shape (2, 2)"
    
    # Check orthonormality
    VTV = V.T @ V
    np.testing.assert_allclose(VTV, np.eye(2), atol=1e-10)

