"""
MATLAB I/O pairs from debug_matlab_outputSet.m and debug_matlab_outputSet_generated.m
This test has been verified against MATLAB execution.

Source: cora_matlab/contDynamics/@linearSys/outputSet.m
Verified: 2025-01-XX

Tests the outputSet method for linear systems.
All generated tests (test_02 through test_05) have been verified with MATLAB I/O pairs.
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
# Note: outputSet expects single sets, not ReachSet objects


def test_outputSet_01_basic():
    """
    MATLAB I/O pairs from debug_matlab_outputSet.m
    MATLAB tolerance: 1e-14
    
    Tests the core functionality of outputSet.
    Verifies that:
    1. Output set is computed correctly
    2. Verror is always 0 for linear systems
    3. Handles output equation with C matrix
    """
    # Setup: Create a linear system with output equation (matching MATLAB test)
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    C = np.array([[1, 0]], dtype=float)  # Output: y = x1
    sys = LinearSys(A, B, None, C)  # Fixed: removed incorrect 'test_sys' argument
    
    # Create a reachable set (single set, not ReachSet)
    # outputSet expects a single set, not a ReachSet object
    R = Zonotope(np.array([[1], [1]], dtype=float), 0.1 * np.eye(2, dtype=float))
    
    params = {
        'U': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0)),
        'uTrans': np.zeros((1, 1), dtype=float),
        'V': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0))
    }
    
    options = {
        'compOutputSet': True
    }
    
    tol = 1e-14  # MATLAB tolerance
    
    # Execute
    # Note: Python outputSet returns (Y, Verror) where Verror is always 0
    # MATLAB only returns Y, but Verror is always 0 for linear systems
    result = sys.outputSet(R, params, options)
    if isinstance(result, tuple):
        Y, Verror = result
    else:
        Y = result
        Verror = 0
    
    # Verify
    # 1. Y should be computed and be a Zonotope (matching MATLAB)
    assert Y is not None, "Y should be computed"
    assert isinstance(Y, Zonotope), f"Y should be a Zonotope, got {type(Y)}"
    
    # 2. Verror should be 0 for linear systems (matching MATLAB)
    assert Verror == 0, "Verror should be 0 for linear systems"
    
    # 3. MATLAB values: Y center = [1], Y.G = [0.1, 0]
    Y_center = Y.center()
    Y_generators = Y.generators()
    Y_center_matlab = np.array([[1]], dtype=float)
    Y_G_matlab = np.array([[0.1, 0]], dtype=float)
    
    np.testing.assert_allclose(Y_center, Y_center_matlab, atol=tol, 
                               err_msg="Y.center() should match MATLAB")
    assert Y_generators.shape == (1, 2), f"Y.generators() should have shape (1, 2), got {Y_generators.shape}"
    np.testing.assert_allclose(Y_generators, Y_G_matlab, atol=tol,
                               err_msg="Y.generators() should match MATLAB")


def test_outputSet_02_no_output_equation():
    """
    MATLAB I/O pairs from debug_matlab_outputSet_generated.m
    This test has been verified against MATLAB execution.
    
    Tests outputSet when output equation is empty (y = x).
    MATLAB Output:
    - Y.c = [1; 1]
    - Y.G = [[0.1, 0], [0, 0.1]]
    - Y should equal R (no output equation)
    """
    # Setup: System without explicit output equation
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    sys = LinearSys(A, B)  # No C matrix
    
    # outputSet expects a single set, not a ReachSet object
    R = Zonotope(np.array([[1], [1]], dtype=float), 0.1 * np.eye(2, dtype=float))
    
    params = {
        'U': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0)),
        'uTrans': np.zeros((1, 1), dtype=float),
        'V': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0))
    }
    
    options = {
        'compOutputSet': True
    }
    
    tol = 1e-14  # MATLAB tolerance
    
    # Execute
    result = sys.outputSet(R, params, options)
    if isinstance(result, tuple):
        Y, Verror = result
    else:
        Y = result
        Verror = 0
    
    # Verify
    assert Y is not None, "Y should be computed even without output equation"
    assert Verror == 0, "Verror should be 0"
    
    # MATLAB values: Y.c = [1; 1], Y.G = [[0.1, 0], [0, 0.1]]
    Y_center = Y.center()
    Y_generators = Y.generators()
    Y_center_matlab = np.array([[1], [1]], dtype=float)
    Y_G_matlab = np.array([[0.1, 0], [0, 0.1]], dtype=float)
    
    np.testing.assert_allclose(Y_center, Y_center_matlab, atol=tol,
                               err_msg="Y.center() should match MATLAB")
    np.testing.assert_allclose(Y_generators, Y_G_matlab, atol=tol,
                               err_msg="Y.generators() should match MATLAB")


def test_outputSet_03_with_D_matrix():
    """
    MATLAB I/O pairs from debug_matlab_outputSet_generated.m
    This test has been verified against MATLAB execution.
    
    Tests outputSet when D matrix is present (direct feedthrough).
    MATLAB Output:
    - Y.c = [1.1]
    - Y.G = [[0.1, 0, 0.025]]
    """
    # Setup: System with D matrix
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    C = np.array([[1, 0]], dtype=float)
    D = np.array([[0.5]], dtype=float)  # Direct feedthrough
    sys = LinearSys(A, B, None, C, D)
    
    # outputSet expects a single set, not a ReachSet object
    R = Zonotope(np.array([[1], [1]], dtype=float), 0.1 * np.eye(2, dtype=float))
    
    params = {
        'U': Zonotope(np.array([[0.1]], dtype=float), 0.05 * np.array([[1]], dtype=float)),
        'uTrans': np.array([[0.1]], dtype=float),
        'V': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0))
    }
    
    options = {
        'compOutputSet': True
    }
    
    tol = 1e-14  # MATLAB tolerance
    
    # Execute
    result = sys.outputSet(R, params, options)
    if isinstance(result, tuple):
        Y, Verror = result
    else:
        Y = result
        Verror = 0
    
    # Verify
    assert Y is not None, "Y should be computed with D matrix"
    assert Verror == 0, "Verror should be 0"
    
    # MATLAB values: Y.c = [1.1], Y.G = [[0.1, 0, 0.025]]
    Y_center = Y.center()
    Y_generators = Y.generators()
    Y_center_matlab = np.array([[1.1]], dtype=float)
    Y_G_matlab = np.array([[0.1, 0, 0.025]], dtype=float)
    
    np.testing.assert_allclose(Y_center, Y_center_matlab, atol=tol,
                               err_msg="Y.center() should match MATLAB")
    np.testing.assert_allclose(Y_generators, Y_G_matlab, atol=tol,
                               err_msg="Y.generators() should match MATLAB")


def test_outputSet_04_compOutputSet_false():
    """
    MATLAB I/O pairs from debug_matlab_outputSet_generated.m
    This test has been verified against MATLAB execution.
    
    Tests that outputSet skips computation when compOutputSet is False.
    MATLAB Output:
    - Y should equal R (compOutputSet = false)
    - Y.c = [1; 1]
    - Y.G = [[0.1, 0], [0, 0.1]]
    """
    # Setup
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    C = np.array([[1, 0]], dtype=float)
    sys = LinearSys(A, B, None, C)
    
    # outputSet expects a single set, not a ReachSet object
    R = Zonotope(np.array([[1], [1]], dtype=float), 0.1 * np.eye(2, dtype=float))
    
    params = {
        'U': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0)),
        'uTrans': np.zeros((1, 1), dtype=float),
        'V': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0))
    }
    
    options = {
        'compOutputSet': False  # Skip computation
    }
    
    tol = 1e-14  # MATLAB tolerance
    
    # Execute
    result = sys.outputSet(R, params, options)
    if isinstance(result, tuple):
        Y, Verror = result
    else:
        Y = result
        Verror = 0
    
    # Verify
    # When compOutputSet is False, Y should equal R (MATLAB behavior)
    assert Verror == 0, "Verror should be 0"
    
    # MATLAB values: Y should equal R, Y.c = [1; 1], Y.G = [[0.1, 0], [0, 0.1]]
    Y_center = Y.center()
    Y_generators = Y.generators()
    Y_center_matlab = np.array([[1], [1]], dtype=float)
    Y_G_matlab = np.array([[0.1, 0], [0, 0.1]], dtype=float)
    
    np.testing.assert_allclose(Y_center, Y_center_matlab, atol=tol,
                               err_msg="Y.center() should match MATLAB (should equal R)")
    np.testing.assert_allclose(Y_generators, Y_G_matlab, atol=tol,
                               err_msg="Y.generators() should match MATLAB (should equal R)")


def test_outputSet_05_with_reduction():
    """
    MATLAB I/O pairs from debug_matlab_outputSet_generated.m
    This test has been verified against MATLAB execution.
    
    Tests outputSet with order reduction (saveOrder option).
    MATLAB Output:
    - Y.c = [1]
    - Y.G = [[0.1]]
    - Y should have reduced order (saveOrder = 5)
    """
    # Setup
    A = np.array([[-1, 0],
                  [0, -2]], dtype=float)
    B = np.array([[1], [1]], dtype=float)
    C = np.array([[1, 0]], dtype=float)
    sys = LinearSys(A, B, None, C)
    
    # outputSet expects a single set, not a ReachSet object
    R = Zonotope(np.array([[1], [1]], dtype=float), 0.1 * np.eye(2, dtype=float))
    
    params = {
        'U': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0)),
        'uTrans': np.zeros((1, 1), dtype=float),
        'V': Zonotope(np.zeros((1, 1), dtype=float), np.array([]).reshape(1, 0))
    }
    
    options = {
        'compOutputSet': True,
        'saveOrder': 5,
        'reductionTechnique': 'girard'
    }
    
    tol = 1e-14  # MATLAB tolerance
    
    # Execute
    result = sys.outputSet(R, params, options)
    if isinstance(result, tuple):
        Y, Verror = result
    else:
        Y = result
        Verror = 0
    
    # Verify
    assert Y is not None, "Y should be computed with order reduction"
    assert Verror == 0, "Verror should be 0"
    
    # MATLAB values: Y.c = [1], Y.G = [[0.1]]
    Y_center = Y.center()
    Y_generators = Y.generators()
    Y_center_matlab = np.array([[1]], dtype=float)
    Y_G_matlab = np.array([[0.1]], dtype=float)
    
    np.testing.assert_allclose(Y_center, Y_center_matlab, atol=tol,
                               err_msg="Y.center() should match MATLAB")
    np.testing.assert_allclose(Y_generators, Y_G_matlab, atol=tol,
                               err_msg="Y.generators() should match MATLAB")
    # Verify order reduction: should have fewer generators than original
    assert Y_generators.shape[1] <= options['saveOrder'], \
        f"Y should have order <= {options['saveOrder']}, got {Y_generators.shape[1]}"

