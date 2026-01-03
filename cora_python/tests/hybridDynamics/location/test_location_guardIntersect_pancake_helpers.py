"""
test_location_guardIntersect_pancake_helpers - test function for guardIntersect_pancake helper methods

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/guardIntersect_pancake.m (auxiliary functions)
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.guardIntersect_pancake import (
    _aux_scaledSystem, _aux_reachTimeScaled, _aux_jump,
    _aux_defaultOptions, _aux_dynamicsLinSys
)
from cora_python.hybridDynamics.location.location import Location
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope.zonotope import Zonotope


def test_aux_scaledSystem_01_basic():
    """
    GENERATED TEST - Basic aux_scaledSystem test
    
    Tests scaled system computation.
    """
    # create system
    sys = LinearSys('linearSys', np.array([[0, 1], [-1, 0]]), np.array([[1]]))
    
    # create polytope
    P = Polytope(np.array([[1, 0]]), np.array([[1]]))
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # guard ID
    guardID = 0
    
    # params
    params = {
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    }
    
    sys_scaled, params_scaled = _aux_scaledSystem(sys, P, R0, guardID, params)
    
    assert sys_scaled is not None, "Should return scaled system"
    assert isinstance(sys_scaled, LinearSys), "Should be a LinearSys"
    assert params_scaled is not None, "Should return scaled params"
    assert isinstance(params_scaled, dict), "Should be a dictionary"


def test_aux_reachTimeScaled_01_basic():
    """
    GENERATED TEST - Basic aux_reachTimeScaled test
    
    Tests reach time computation for scaled system.
    """
    # create location
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    from cora_python.hybridDynamics.transition.transition import Transition
    from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
    reset = LinearReset(np.eye(2))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [0]]))
    loc = Location(inv, [trans], flow)
    
    # create polytope
    P = Polytope(np.array([[1, 0]]), np.array([[1]]))
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # params
    params = {
        'R0': R0,
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)),
        'W': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    }
    
    t = _aux_reachTimeScaled(loc, P, R0, params)
    
    assert isinstance(t, (int, float, np.number)), "Should return time value"
    assert t >= 0, "Time should be non-negative"


def test_aux_jump_01_basic():
    """
    GENERATED TEST - Basic aux_jump test
    
    Tests jump computation for pancake method.
    """
    # create location
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    from cora_python.hybridDynamics.transition.transition import Transition
    from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
    reset = LinearReset(np.eye(2))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [0]]))
    loc = Location(inv, [trans], flow)
    
    # create polytope
    P = Polytope(np.array([[1, 0]]), np.array([[1]]))
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # params
    params = {
        'R0': R0,
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)),
        'W': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    }
    
    R_jump = _aux_jump(loc, P, R0, params)
    
    assert R_jump is not None, "Should return jump set"
    assert hasattr(R_jump, 'center') or hasattr(R_jump, 'c'), \
        "Should be a set with center"


def test_aux_defaultOptions_01_basic():
    """
    GENERATED TEST - Basic aux_defaultOptions test
    
    Tests default options setting.
    """
    # create options dict
    options = {
        'timeStep': 0.1
    }
    
    options_default = _aux_defaultOptions(options)
    
    assert isinstance(options_default, dict), "Should return dictionary"
    assert 'timeStep' in options_default, "Should preserve existing options"
    # Should add default values for missing options


def test_aux_dynamicsLinSys_01_basic():
    """
    GENERATED TEST - Basic aux_dynamicsLinSys test
    
    Tests linear system dynamics evaluation.
    """
    # create system
    sys = LinearSys('linearSys', np.array([[0, 1], [-1, 0]]), np.array([[1]]))
    
    # state
    x = np.array([[1], [0]])
    
    # input
    u = np.array([[0.5]])
    
    dx = _aux_dynamicsLinSys(x, u, sys)
    
    assert dx is not None, "Should return derivative"
    assert isinstance(dx, np.ndarray), "Should return numpy array"
    assert dx.shape == x.shape, "Should have same shape as state"
    # dx = A*x + B*u + c
    dx_expected = sys.A @ x + sys.B @ u + (sys.c if hasattr(sys, 'c') and sys.c is not None else 0)
    assert np.allclose(dx, dx_expected, atol=1e-6), \
        "Should match expected dynamics"

