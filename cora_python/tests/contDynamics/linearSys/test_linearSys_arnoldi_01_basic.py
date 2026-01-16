"""
GENERATED TEST - Additional test cases beyond MATLAB test
This test is generated based on MATLAB source code logic.

Source: cora_matlab/global/functions/helper/dynamics/contDynamics/linearSys/arnoldi.m
Generated: 2025-01-XX

Note: There IS a MATLAB test file: test_Krylov_Arnoldi.m
      It has been translated to: test_linearSys_arnoldi_02_matlab_test.py
      This file contains additional generated test cases for broader coverage.

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
    
    Verified against MATLAB using debug_matlab_arnoldi_generated.m
    MATLAB values extracted for comparison (seed 42, 5x5 matrix, redDim=3)
    """
    # Setup: Use exact values from MATLAB (rng(42), then randn)
    # MATLAB generates different values than Python's randn, so we use exact MATLAB values
    # Extracted from MATLAB: rng(42); A = randn(5,5); A = (A + A.')/2; vInit = randn(5,1); vInit = vInit/norm(vInit);
    A = np.array([
        [-0.53824389372927, 0.172009062971566, 0.1922432518503, -0.171757813944017, -0.702575971202734],
        [0.172009062971566, -1.29744747716739, 1.38088911307322, 0.0953854775075922, -0.263337443603668],
        [0.1922432518503, 1.38088911307322, 1.81692224917996, -0.0916919803691555, -0.697752092261864],
        [-0.171757813944017, 0.0953854775075922, -0.0916919803691555, -0.661011014460339, 0.120582029588883],
        [-0.702575971202734, -0.263337443603668, -0.697752092261864, 0.120582029588883, 1.00038318883897]
    ])
    
    # Exact vInit from MATLAB (already normalized)
    vInit = np.array([[-0.343554269032129],
                      [0.039357461662057],
                      [0.117388451102307],
                      [-0.914682502467213],
                      [0.173197360457476]])
    
    # Reduced dimension
    redDim = 3
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    # 1. V should have correct dimensions
    assert V.shape == (5, redDim), f"V should have shape (5, {redDim}), got {V.shape}"
    
    # 2. V should be orthonormal (columns are orthonormal)
    # MATLAB: VTV(1,1) = 1, VTV(1,2) ≈ 5.83e-16, VTV(2,2) = 1
    VTV = V.T @ V
    np.testing.assert_allclose(VTV, np.eye(redDim), atol=1e-10,
                               err_msg="V should be orthonormal")
    
    # MATLAB verified values (from debug_matlab_arnoldi_generated.m):
    # V(1,1) = -0.343554269032129, V(2,1) = 0.039357461662057, V(1,2) = 0.0610428370258458
    # H(1,1) = -0.652603500864434, H(1,2) = 0.41721209267439, H(2,1) = 0.41721209267439
    # Hlast = 0.953933887734142, happyBreakdown = 0
    np.testing.assert_allclose(V[0, 0], -0.343554269032129, atol=1e-10, err_msg="V(1,1) should match MATLAB")
    np.testing.assert_allclose(V[1, 0], 0.039357461662057, atol=1e-10, err_msg="V(2,1) should match MATLAB")
    np.testing.assert_allclose(V[0, 1], 0.0610428370258458, atol=1e-10, err_msg="V(1,2) should match MATLAB")
    np.testing.assert_allclose(H[0, 0], -0.652603500864434, atol=1e-10, err_msg="H(1,1) should match MATLAB")
    np.testing.assert_allclose(H[0, 1], 0.41721209267439, atol=1e-10, err_msg="H(1,2) should match MATLAB")
    np.testing.assert_allclose(H[1, 0], 0.41721209267439, atol=1e-10, err_msg="H(2,1) should match MATLAB")
    np.testing.assert_allclose(Hlast, 0.953933887734142, atol=1e-10, err_msg="Hlast should match MATLAB")
    assert happyBreakdown == False, "happyBreakdown should be False (MATLAB: 0)"
    
    # 3. H should be upper Hessenberg (zeros below first subdiagonal)
    for i in range(H.shape[0]):
        for j in range(i - 1):  # j < i-1 should be zero
            assert abs(H[i, j]) < 1e-10, f"H[{i},{j}] should be zero (upper Hessenberg)"
    
    # 4. Hlast should be the last computed H(j+1,j) value
    if not happyBreakdown:
        # Hlast should be a positive value (norm of a vector)
        assert Hlast > 0, "Hlast should be positive"
        assert np.isfinite(Hlast), "Hlast should be finite"
    
    # 5. Arnoldi relation: A*V(:,1:j) = V*H + H(j+1,j)*V(:,j+1)*e_j^T
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
    
    Verified against MATLAB using debug_matlab_arnoldi_generated.m
    MATLAB: A2 = eye(3), vInit2 = [1; 0; 0], redDim2 = 5
    MATLAB output: V2 shape = 3x1, H2 shape = 1x1, happyBreakdown2 = 1
    """
    # Setup: Create a matrix that will cause early termination
    # MATLAB: A2 = eye(3), vInit2 = [1; 0; 0], redDim2 = 5
    A = np.eye(3)
    vInit = np.array([[1], [0], [0]])
    redDim = 5  # Request more dimensions than possible
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    # MATLAB verified: V2 shape = 3x1, H2 shape = 1x1, happyBreakdown2 = 1
    assert V.shape == (3, 1), f"V should have shape (3, 1), got {V.shape}"
    assert H.shape == (1, 1), f"H should have shape (1, 1), got {H.shape}"
    assert happyBreakdown == True, "happyBreakdown should be True (MATLAB: 1)"
    
    # V should still be orthonormal
    VTV = V.T @ V
    np.testing.assert_allclose(VTV, np.eye(1), atol=1e-10,
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
    MATLAB I/O pairs from debug_matlab_arnoldi.m
    This test has been verified against MATLAB execution.
    
    Tests arnoldi on a 2x2 system.
    MATLAB shows happyBreakdown = 1 (true) for this case because the Krylov
    subspace is exhausted (Hlast = 0, meaning the next vector has zero norm).
    """
    # Setup: Create a 2x2 system (matching MATLAB test)
    A = np.array([[1, 2],
                  [3, 4]], dtype=float)
    vInit = np.array([[1], [0]], dtype=float)
    redDim = 2
    
    # Execute
    V, H, Hlast, happyBreakdown = arnoldi(A, vInit, redDim)
    
    # Verify
    assert V.shape == (2, 2), "V should have shape (2, 2)"
    assert H.shape == (2, 2), "H should have shape (2, 2)"
    # MATLAB: happyBreakdown = 1 (true) for this case
    assert happyBreakdown, "Should have happy breakdown when Krylov subspace is exhausted"
    # MATLAB: Hlast = 0 (next vector has zero norm)
    assert abs(Hlast) < 1e-14, f"Hlast should be approximately 0, got {Hlast}"
    
    # MATLAB values: V = [1, 0; 0, 1], H = [1, 2; 3, 4]
    V_matlab = np.array([[1, 0], [0, 1]], dtype=float)
    H_matlab = np.array([[1, 2], [3, 4]], dtype=float)
    np.testing.assert_allclose(V, V_matlab, atol=1e-14, err_msg="V should match MATLAB")
    np.testing.assert_allclose(H, H_matlab, atol=1e-14, err_msg="H should match MATLAB")
    
    # Check orthonormality
    VTV = V.T @ V
    np.testing.assert_allclose(VTV, np.eye(2), atol=1e-10, err_msg="V should be orthonormal")


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

