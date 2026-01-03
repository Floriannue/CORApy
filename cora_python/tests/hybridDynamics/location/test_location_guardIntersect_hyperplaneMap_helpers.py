"""
test_location_guardIntersect_hyperplaneMap_helpers - test function for guardIntersect_hyperplaneMap helper methods

GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/hybridDynamics/@location/guardIntersect_hyperplaneMap.m (auxiliary functions)
Generated: 2025-01-XX
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.guardIntersect_hyperplaneMap import (
    _aux_refinedIntersectionTime, _aux_abstractionError,
    _aux_systemParams, _aux_mappedSetError, _aux_constantFlow,
    _aux_taylorSeriesParam
)
from cora_python.hybridDynamics.location.location import Location
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval


def test_aux_refinedIntersectionTime_01_basic():
    """
    GENERATED TEST - Basic aux_refinedIntersectionTime test
    
    Tests refined intersection time computation.
    """
    # create location
    inv = Interval(np.array([[-2], [-2]]), np.array([[2], [2]]))
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    from cora_python.hybridDynamics.transition.transition import Transition
    from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
    reset = LinearReset(np.eye(2))
    trans = Transition(guard, reset, 1)
    flow = LinearSys('linearSys', np.zeros((2, 2)), np.array([[0], [0]]), np.array([[1], [0]]))
    loc = Location(inv, [trans], flow)
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # params
    params = {
        'R0': R0,
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)),
        'W': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    }
    
    # options
    options = {
        'timeStep': 0.1,
        'taylorTerms': 10
    }
    
    Rmin, tmin, tmax, R = _aux_refinedIntersectionTime(loc, guard, R0, params, options)
    
    assert Rmin is not None, "Should return minimum reachable set"
    assert isinstance(tmin, (int, float, np.number)), "tmin should be a number"
    assert isinstance(tmax, (int, float, np.number)), "tmax should be a number"
    assert tmin <= tmax, "tmin should be <= tmax"
    assert R is not None, "Should return reachable set"


def test_aux_abstractionError_01_basic():
    """
    GENERATED TEST - Basic aux_abstractionError test
    
    Tests abstraction error computation.
    """
    # create system matrix
    A = np.array([[0, 1], [-1, 0]])
    
    # create input set
    U = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # time parameters
    th = 0.1
    tmax = 0.2
    order = 10
    
    err = _aux_abstractionError(A, U, R0, th, tmax, order)
    
    assert err is not None, "Should return error set"
    assert hasattr(err, 'center') or hasattr(err, 'c'), \
        "Should be a set with center"


def test_aux_systemParams_01_basic():
    """
    GENERATED TEST - Basic aux_systemParams test
    
    Tests system parameter extraction.
    """
    # create system
    sys = LinearSys('linearSys', np.array([[0, 1], [-1, 0]]), np.array([[1]]))
    
    # create reachable set (dummy)
    class DummyRcont:
        pass
    Rcont = DummyRcont()
    
    # params
    params = {
        'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    }
    
    A, U = _aux_systemParams(sys, Rcont, params)
    
    assert A is not None, "Should return system matrix"
    assert isinstance(A, np.ndarray), "A should be numpy array"
    assert U is not None, "Should return input set"


def test_aux_mappedSetError_01_basic():
    """
    GENERATED TEST - Basic aux_mappedSetError test
    
    Tests mapped set error computation.
    """
    # create guard
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # system matrix and offset
    A = np.array([[0, 1], [-1, 0]])
    b = np.array([[1], [0]])
    
    # error set
    err = Zonotope(np.zeros((2, 1)), 0.01 * np.eye(2))
    
    R_err = _aux_mappedSetError(guard, R0, A, b, err)
    
    assert R_err is not None, "Should return error set"
    assert hasattr(R_err, 'center') or hasattr(R_err, 'c'), \
        "Should be a set with center"


def test_aux_constantFlow_01_basic():
    """
    GENERATED TEST - Basic aux_constantFlow test
    
    Tests constant flow computation.
    """
    # system matrix
    A = np.array([[0, 1], [-1, 0]])
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    # input set
    U = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    
    # time
    th = 0.1
    
    # order
    order = 10
    
    R_flow = _aux_constantFlow(A, R0, U, th, order)
    
    assert R_flow is not None, "Should return flow set"
    assert isinstance(R_flow, np.ndarray), "Should return numpy array"
    assert R_flow.shape[0] == R0.dim(), "Should have correct dimension"


def test_aux_taylorSeriesParam_01_basic():
    """
    GENERATED TEST - Basic aux_taylorSeriesParam test
    
    Tests Taylor series parameter computation.
    """
    # create guard
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[1, 0]]), np.array([[1]]))
    
    # system matrix and offset
    A = np.array([[0, 1], [-1, 0]])
    b = np.array([[1], [0]])
    
    # initial set
    R0 = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
    
    alpha, beta, gamma_list, I = _aux_taylorSeriesParam(guard, A, b, R0)
    
    assert alpha is not None, "Should return alpha"
    assert isinstance(alpha, np.ndarray), "alpha should be numpy array"
    assert beta is not None, "Should return beta"
    assert isinstance(beta, np.ndarray), "beta should be numpy array"
    assert isinstance(gamma_list, list), "gamma_list should be a list"
    assert I is not None, "Should return interval"
    assert isinstance(I, Interval), "I should be an Interval"

