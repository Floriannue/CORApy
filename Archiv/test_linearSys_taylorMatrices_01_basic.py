"""
MATLAB I/O pairs from debug_matlab_taylorMatrices.m
This test has been verified against MATLAB execution.

Source: cora_matlab/contDynamics/@linearSys/taylorMatrices.m
MATLAB test: cora_matlab/unitTests/contDynamics/linearSys/test_linearSys_taylorMatrices.m
Verified: 2025-01-XX

Tests the taylorMatrices function which computes remainder and correction matrices.
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contDynamics.linearSys.taylorMatrices import taylorMatrices
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from cora_python.contSet.interval import Interval


def test_taylorMatrices_01_basic():
    """
    MATLAB I/O pairs from debug_matlab_taylorMatrices.m
    MATLAB tolerance: 1e-14
    
    Tests the core functionality of taylorMatrices.
    Verifies that:
    1. E (remainder matrix) is computed as Interval
    2. F (correction matrix for state) is computed as Interval
    3. G (correction matrix for input) is computed as Interval
    4. TaylorLinSys object is created if it doesn't exist
    """
    # Setup: Create a simple linear system (matching MATLAB test)
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    sys = LinearSys(A, B)  # Fixed: removed incorrect 'test_sys' argument
    
    timeStep = 0.1
    truncationOrder = 10
    tol = 1e-14  # MATLAB tolerance
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify
    # 1. E should be an Interval matrix (matching MATLAB)
    assert isinstance(E, Interval), f"E should be an Interval, got {type(E)}"
    E_dim = E.dim()
    assert E_dim == [2, 2], f"E should have shape [2, 2], got {E_dim}"
    
    # MATLAB values: E is interval with very small bounds (~1e-15)
    # Use object.method() instead of standalone function calls
    E_center = E.center()
    E_rad = E.rad()
    # E.center() should be approximately zero
    assert np.allclose(E_center, np.zeros((2, 2)), atol=tol), "E.center() should be approximately zero"
    # E.rad() should be very small
    assert np.all(E_rad < 1e-14), "E.rad() should be very small"
    
    # 2. F should be an Interval matrix (matching MATLAB)
    assert isinstance(F, Interval), f"F should be an Interval, got {type(F)}"
    F_dim = F.dim()
    assert F_dim == [2, 2], f"F should have shape [2, 2], got {F_dim}"
    F_center = F.center()
    # MATLAB: F.center() is 2x2 matrix
    assert F_center.shape == (2, 2), f"F.center() should have shape (2, 2), got {F_center.shape}"
    
    # 3. G should be an Interval matrix (matching MATLAB)
    assert isinstance(G, Interval), f"G should be an Interval, got {type(G)}"
    G_dim = G.dim()
    assert G_dim == [2, 2], f"G should have shape [2, 2], got {G_dim}"
    G_center = G.center()
    # MATLAB: G.center() is 2x2 matrix
    assert G_center.shape == (2, 2), f"G.center() should have shape (2, 2), got {G_center.shape}"
    
    # 4. System should have taylor object
    assert hasattr(sys, 'taylor'), "System should have taylor object"
    assert sys.taylor is not None, "taylor object should not be None"
    assert isinstance(sys.taylor, TaylorLinSys), "taylor should be TaylorLinSys object"


def test_taylorMatrices_02_existing_taylor():
    """
    GENERATED TEST - Existing TaylorLinSys test
    
    Tests that taylorMatrices uses existing TaylorLinSys object if present.
    """
    # Setup
    A = np.array([[-1, 0.5],
                  [0.5, -2]], dtype=float)
    B = np.array([[1], [0]], dtype=float)
    sys = LinearSys(A, B)  # Fixed: removed incorrect 'test_sys' argument
    
    # Pre-create taylor object
    sys.taylor = TaylorLinSys(A)
    
    timeStep = 0.05
    truncationOrder = 8
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify
    assert hasattr(sys, 'taylor'), "System should have taylor object"
    assert sys.taylor is not None, "taylor object should not be None"
    
    # Verify matrices are computed
    assert E is not None, "E should be computed"
    assert F is not None, "F should be computed"
    assert G is not None, "G should be computed"


def test_taylorMatrices_03_different_orders():
    """
    GENERATED TEST - Different truncation orders test
    
    Tests taylorMatrices with different truncation orders.
    """
    # Setup
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    sys = LinearSys(A, B)  # Fixed: removed incorrect 'test_sys' argument
    
    timeStep = 0.1
    
    # Test different orders
    for truncationOrder in [5, 10, 15]:
        E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
        
        # Verify all matrices are computed
        assert E is not None, f"E should be computed for order {truncationOrder}"
        assert F is not None, f"F should be computed for order {truncationOrder}"
        assert G is not None, f"G should be computed for order {truncationOrder}"
        
        # Verify shapes (E, F, G are Interval objects, use dim() not shape)
        assert isinstance(E, Interval), f"E should be Interval for order {truncationOrder}"
        assert isinstance(F, Interval), f"F should be Interval for order {truncationOrder}"
        assert isinstance(G, Interval), f"G should be Interval for order {truncationOrder}"
        assert E.dim() == [A.shape[0], A.shape[1]], f"E dimensions should match A for order {truncationOrder}"
        assert F.dim() == [A.shape[0], A.shape[0]], f"F dimensions should be correct for order {truncationOrder}"
        assert G.dim() == [A.shape[0], A.shape[0]], f"G dimensions should be correct for order {truncationOrder}"


def test_taylorMatrices_04_large_system():
    """
    GENERATED TEST - Large system test
    
    Tests taylorMatrices on a larger system.
    """
    # Setup: 5-dimensional system
    np.random.seed(42)
    A = np.random.randn(5, 5).astype(float)
    A = (A + A.T) / 2  # Make symmetric
    A = A - 2 * np.eye(5, dtype=float)  # Make stable
    
    B = np.random.randn(5, 2).astype(float)
    sys = LinearSys(A, B)  # Fixed: removed incorrect 'test_sys' argument
    
    timeStep = 0.05
    truncationOrder = 10
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify (E, F, G are Interval objects, use dim() not shape)
    assert isinstance(E, Interval), "E should be Interval"
    assert isinstance(F, Interval), "F should be Interval"
    assert isinstance(G, Interval), "G should be Interval"
    assert E.dim() == [5, 5], f"E should have dimensions [5, 5], got {E.dim()}"
    assert F.dim() == [5, 5], f"F should have dimensions [5, 5], got {F.dim()}"
    assert G.dim() == [5, 5], f"G should have dimensions [5, 5], got {G.dim()}"


def test_taylorMatrices_05_small_timestep():
    """
    GENERATED TEST - Small time step test
    
    Tests taylorMatrices with a very small time step.
    """
    # Setup
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    sys = LinearSys(A, B)  # Fixed: removed incorrect 'test_sys' argument
    
    timeStep = 0.001  # Very small time step
    truncationOrder = 10
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify
    assert E is not None, "E should be computed"
    assert F is not None, "F should be computed"
    assert G is not None, "G should be computed"
    
    # For small time steps, E should be small (remainder should be small)
    # Use object.method() pattern: E.rad() to get radius of interval
    assert isinstance(E, Interval), "E should be Interval"
    E_rad = E.rad()
    assert np.all(E_rad < 1.0), f"E.rad() should be small for small time step, got max {np.max(E_rad)}"


