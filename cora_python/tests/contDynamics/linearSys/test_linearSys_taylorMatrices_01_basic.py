"""
GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/contDynamics/@linearSys/taylorMatrices.m
Generated: 2025-01-XX

Tests the taylorMatrices function which computes remainder and correction matrices.
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contDynamics.linearSys.taylorMatrices import taylorMatrices
from cora_python.g.classes.taylorLinSys import TaylorLinSys


def test_taylorMatrices_01_basic():
    """
    GENERATED TEST - Basic functionality test
    
    Tests the core functionality of taylorMatrices based on MATLAB source code.
    Verifies that:
    1. E (remainder matrix) is computed
    2. F (correction matrix for state) is computed
    3. G (correction matrix for input) is computed
    4. TaylorLinSys object is created if it doesn't exist
    """
    # Setup: Create a simple linear system
    A = np.array([[-1, 0],
                  [0, -2]])
    B = np.array([[1], [1]])
    sys = LinearSys('test_sys', A, B)
    
    timeStep = 0.1
    truncationOrder = 10
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify
    # 1. E should be a matrix
    assert isinstance(E, np.ndarray), "E should be a numpy array"
    assert E.shape == A.shape, f"E should have shape {A.shape}, got {E.shape}"
    
    # 2. F should be a matrix
    assert isinstance(F, np.ndarray), "F should be a numpy array"
    assert F.shape[0] == A.shape[0], f"F should have {A.shape[0]} rows"
    
    # 3. G should be a matrix
    assert isinstance(G, np.ndarray), "G should be a numpy array"
    assert G.shape[0] == A.shape[0], f"G should have {A.shape[0]} rows"
    assert G.shape[1] == B.shape[1], f"G should have {B.shape[1]} columns"
    
    # 4. System should have taylor object
    assert hasattr(sys, 'taylor'), "System should have taylor object"
    assert isinstance(sys.taylor, TaylorLinSys), "taylor should be TaylorLinSys object"


def test_taylorMatrices_02_existing_taylor():
    """
    GENERATED TEST - Existing TaylorLinSys test
    
    Tests that taylorMatrices uses existing TaylorLinSys object if present.
    """
    # Setup
    A = np.array([[-1, 0.5],
                  [0.5, -2]])
    B = np.array([[1], [0]])
    sys = LinearSys('test_sys', A, B)
    
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
                  [0, -2]])
    B = np.array([[1], [1]])
    sys = LinearSys('test_sys', A, B)
    
    timeStep = 0.1
    
    # Test different orders
    for truncationOrder in [5, 10, 15]:
        E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
        
        # Verify all matrices are computed
        assert E is not None, f"E should be computed for order {truncationOrder}"
        assert F is not None, f"F should be computed for order {truncationOrder}"
        assert G is not None, f"G should be computed for order {truncationOrder}"
        
        # Verify shapes
        assert E.shape == A.shape, f"E shape should match A for order {truncationOrder}"
        assert F.shape[0] == A.shape[0], f"F should have correct rows for order {truncationOrder}"
        assert G.shape[0] == A.shape[0], f"G should have correct rows for order {truncationOrder}"


def test_taylorMatrices_04_large_system():
    """
    GENERATED TEST - Large system test
    
    Tests taylorMatrices on a larger system.
    """
    # Setup: 5-dimensional system
    np.random.seed(42)
    A = np.random.randn(5, 5)
    A = (A + A.T) / 2  # Make symmetric
    A = A - 2 * np.eye(5)  # Make stable
    
    B = np.random.randn(5, 2)
    sys = LinearSys('test_sys', A, B)
    
    timeStep = 0.05
    truncationOrder = 10
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify
    assert E.shape == (5, 5), "E should have shape (5, 5)"
    assert F.shape[0] == 5, "F should have 5 rows"
    assert G.shape == (5, 2), "G should have shape (5, 2)"


def test_taylorMatrices_05_small_timestep():
    """
    GENERATED TEST - Small time step test
    
    Tests taylorMatrices with a very small time step.
    """
    # Setup
    A = np.array([[-1, 0],
                  [0, -2]])
    B = np.array([[1], [1]])
    sys = LinearSys('test_sys', A, B)
    
    timeStep = 0.001  # Very small time step
    truncationOrder = 10
    
    # Execute
    E, F, G = taylorMatrices(sys, timeStep, truncationOrder)
    
    # Verify
    # For small time steps, E should be small (remainder should be small)
    assert np.all(np.abs(E) < 1.0), "E should be small for small time step"
    assert E is not None, "E should be computed"
    assert F is not None, "F should be computed"
    assert G is not None, "G should be computed"


