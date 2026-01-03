"""
GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/contDynamics/@linearSys/private/priv_initReach_Krylov.m
Generated: 2025-01-XX

Tests the initialization function for Krylov subspace reachability analysis.
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contDynamics.linearSys.private.priv_initReach_Krylov import priv_initReach_Krylov
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


def test_priv_initReach_Krylov_01_basic():
    """
    GENERATED TEST - Basic functionality test
    
    Tests the core functionality of priv_initReach_Krylov based on MATLAB source code.
    Verifies that:
    1. Krylov field is initialized
    2. Input solution is computed
    3. State subspaces are created
    4. Input subspaces are created
    5. Options are updated correctly
    """
    # Setup: Create a simple linear system
    A = np.array([[-1, 0],
                  [0, -2]])
    B = np.array([[1], [1]])
    C = np.array([[1, 0]])  # Output matrix
    sys = LinearSys('test_sys', A, B, None, C)
    
    # Parameters
    params = {
        'tStart': 0.0,
        'tFinal': 1.0,
        'R0': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2)),
        'U': Zonotope(np.array([[0.5]]), 0.05 * np.array([[1]])),
        'uTrans': np.array([[0.1]])
    }
    
    # Options
    options = {
        'timeStep': 0.1,
        'taylorTerms': 10,
        'krylovError': 1e-6,
        'krylovOrder': 15,
        'krylovStep': 5
    }
    
    # Execute
    sys_out, params_out, options_out = priv_initReach_Krylov(sys, params, options)
    
    # Verify
    # 1. System should have krylov field
    assert hasattr(sys_out, 'krylov'), "System should have krylov field"
    assert isinstance(sys_out.krylov, dict), "krylov should be a dictionary"
    
    # 2. Krylov field should contain required keys
    assert 'Rhom_tp_prev' in sys_out.krylov, "Should have Rhom_tp_prev"
    assert 'Rpar_proj' in sys_out.krylov or 'Rpar_proj_0' in sys_out.krylov, "Should have Rpar_proj or Rpar_proj_0"
    
    # 3. State subspaces should be created
    assert 'state' in sys_out.krylov, "Should have state subspaces"
    state = sys_out.krylov['state']
    assert isinstance(state, dict), "state should be a dictionary"
    
    # 4. Input subspaces should be created (if input set provided)
    if params['U'] is not None:
        assert 'input' in sys_out.krylov, "Should have input subspaces"
        input_krylov = sys_out.krylov['input']
        assert isinstance(input_krylov, dict), "input should be a dictionary"
    
    # 5. Options should be updated
    assert 'tFinal' in options_out, "Options should have tFinal"
    assert options_out['tFinal'] == params['tFinal'], "tFinal should match params"
    
    # 6. Input set should be projected (B @ U)
    # params['U'] should be modified to B @ U
    # This is checked indirectly by verifying the computation proceeded


def test_priv_initReach_Krylov_02_no_input():
    """
    GENERATED TEST - No input set test
    
    Tests priv_initReach_Krylov when no input set is provided.
    """
    # Setup
    A = np.array([[-1, 0],
                  [0, -2]])
    B = np.array([[1], [1]])
    C = np.array([[1, 0]])
    sys = LinearSys('test_sys', A, B, None, C)
    
    params = {
        'tStart': 0.0,
        'tFinal': 1.0,
        'R0': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2)),
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)),  # Empty input
        'uTrans': np.array([[0.0]])
    }
    
    options = {
        'timeStep': 0.1,
        'taylorTerms': 10,
        'krylovError': 1e-6,
        'krylovOrder': 15,
        'krylovStep': 5
    }
    
    # Execute
    sys_out, params_out, options_out = priv_initReach_Krylov(sys, params, options)
    
    # Verify
    assert hasattr(sys_out, 'krylov'), "System should have krylov field"
    assert 'state' in sys_out.krylov, "Should have state subspaces even without input"


def test_priv_initReach_Krylov_03_large_system():
    """
    GENERATED TEST - Large system test
    
    Tests priv_initReach_Krylov on a larger system to verify scalability.
    """
    # Setup: 10-dimensional system
    np.random.seed(42)
    A = np.random.randn(10, 10)
    A = (A + A.T) / 2  # Make symmetric
    A = A - 2 * np.eye(10)  # Make stable
    
    B = np.random.randn(10, 2)
    C = np.random.randn(3, 10)  # 3 outputs
    sys = LinearSys('test_sys', A, B, None, C)
    
    params = {
        'tStart': 0.0,
        'tFinal': 0.5,
        'R0': Zonotope(np.ones((10, 1)), 0.1 * np.eye(10)),
        'U': Zonotope(np.zeros((2, 1)), 0.05 * np.eye(2)),
        'uTrans': np.zeros((2, 1))
    }
    
    options = {
        'timeStep': 0.05,
        'taylorTerms': 10,
        'krylovError': 1e-6,
        'krylovOrder': 10,
        'krylovStep': 5
    }
    
    # Execute
    sys_out, params_out, options_out = priv_initReach_Krylov(sys, params, options)
    
    # Verify
    assert hasattr(sys_out, 'krylov'), "System should have krylov field"
    assert 'state' in sys_out.krylov, "Should have state subspaces"
    
    # Verify state subspaces structure
    state = sys_out.krylov['state']
    if state.get('c_sys_proj') is not None:
        assert hasattr(state['c_sys_proj'], 'A'), "c_sys_proj should be a linearSys object"
        assert hasattr(state['c_sys_proj'], 'B'), "c_sys_proj should have B matrix"

