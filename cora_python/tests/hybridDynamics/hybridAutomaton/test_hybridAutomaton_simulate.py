"""
test_hybridAutomaton_simulate - test function for simulate

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_simulate.m

Authors:       Mark Wetzlinger
Written:       16-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.hybridAutomaton.hybridAutomaton import HybridAutomaton
from cora_python.hybridDynamics.location.location import Location
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


def test_hybridAutomaton_simulate_01_basic():
    """
    TRANSLATED TEST - Basic hybridAutomaton simulate test
    
    Tests simulation with clockwise and counter-clockwise motion.
    """
    # generate simple automaton:
    # - 1st median is guard set
    # - right part moves clockwise
    # - left part moves counter-clockwise
    inv1 = Polytope(np.array([[-1, 1]] / np.sqrt(2)), np.array([[0]]))
    inv2 = Polytope(np.array([[1, -1]] / np.sqrt(2)), np.array([[0]]))
    
    guard = Polytope(np.array([]), np.array([]),
                     np.array([[-1, 1]] / np.sqrt(2)), np.array([[0]]))
    reset1 = LinearReset(np.eye(2), np.zeros((2, 1)), np.array([[-1], [1]]))
    trans1 = Transition(guard, reset1, 2)
    
    reset2 = LinearReset(np.eye(2), np.zeros((2, 1)), np.array([[1], [-1]]))
    trans2 = Transition(guard, reset2, 1)
    
    # clockwise motion
    dyn1 = LinearSys('linearSys', np.array([[0, 1], [-1, 0]]), np.array([[1]]))
    # counter-clockwise motion
    dyn2 = LinearSys('linearSys', np.array([[0, -1], [1, 0]]), np.array([[1]]))
    
    loc1 = Location('clockwise', inv1, [trans1], dyn1)
    loc2 = Location('counter-clockwise', inv2, [trans2], dyn2)
    HA = HybridAutomaton([loc1, loc2])
    
    # model parameters
    params = {
        'x0': np.array([[2], [-2]]),
        'startLoc': 1,
        'finalLoc': 0,
        'tFinal': 3
    }
    
    # simulate trajectory
    t, x, loc = HA.simulate(params)
    
    # must be cell-arrays (lists in Python)
    assert isinstance(t, list) and isinstance(x, list) and isinstance(loc, np.ndarray), \
        "t and x should be lists, loc should be numpy array"
    # must be of same length
    assert len(t) == len(x) and len(t) == len(loc), \
        "t, x, and loc should have same length"
    # check whether points are contained in respective invariant
    x_loc1 = np.vstack([x[i] for i in range(len(x)) if loc[i] == 1])
    x_loc2 = np.vstack([x[i] for i in range(len(x)) if loc[i] == 2])
    assert all(inv1.contains_(x_loc1[i, :].reshape(-1, 1)) for i in range(len(x_loc1))), \
        "Points in loc1 should be in inv1"
    assert all(inv2.contains_(x_loc2[i, :].reshape(-1, 1)) for i in range(len(x_loc2))), \
        "Points in loc2 should be in inv2"
    # time before and after jumps must be the same, but state not
    for i in range(len(t) - 1):
        assert withinTol(t[i][-1], t[i+1][0]), \
            "Time should be continuous across jumps"
        assert not compareMatrices(x[i][-1, :].reshape(-1, 1), x[i+1][0, :].reshape(-1, 1)), \
            "State should change across jumps"


def test_hybridAutomaton_simulate_02_different_dimensions():
    """
    TRANSLATED TEST - Different dimensions per location test
    
    Tests simulation with different number of states per location.
    """
    # automaton with different number of states per location
    inv1 = Polytope(np.array([[-1, 1, 0]] / np.sqrt(2)), np.array([[0]]))
    inv2 = Polytope(np.array([[1, -1, 0]] / np.sqrt(2)), np.array([[0]]))
    
    guard1 = Polytope(np.array([]), np.array([]),
                      np.array([[-1, 1, 0]] / np.sqrt(2)), np.array([[0]]))
    guard2 = Polytope(np.array([]), np.array([]),
                      np.array([[-1, 1, 0]] / np.sqrt(2)), np.array([[0]]))
    reset1 = LinearReset(np.array([[1, 0], [0, 1], [0, 0]]), 
                        np.zeros((3, 1)), np.array([[-1], [1], [1]]))
    trans1 = Transition(guard1, reset1, 2)
    
    reset2 = LinearReset(np.array([[1, 0, 0], [0, 1, 0]]), 
                        np.zeros((2, 1)), np.array([[1], [-1]]))
    trans2 = Transition(guard2, reset2, 1)
    
    # clockwise motion
    dyn1 = LinearSys('linearSys', np.array([[0, 1], [-1, 0]]), np.array([[1]]))
    # counter-clockwise motion
    dyn2 = LinearSys('linearSys', np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]), np.array([[1]]))
    
    loc1 = Location('clockwise', inv1, [trans1], dyn1)
    loc2 = Location('counter-clockwise', inv2, [trans2], dyn2)
    HA = HybridAutomaton([loc1, loc2])
    
    # model parameters
    params = {
        'x0': np.array([[2], [-2]]),
        'startLoc': 1,
        'finalLoc': 0,
        'tFinal': 3
    }
    
    # simulate trajectory
    t, x, loc = HA.simulate(params)
    
    # must be cell-arrays (lists in Python)
    assert isinstance(t, list) and isinstance(x, list) and isinstance(loc, np.ndarray), \
        "t and x should be lists, loc should be numpy array"
    # must be of same length
    assert len(t) == len(x) and len(t) == len(loc), \
        "t, x, and loc should have same length"
    # check whether points are contained in respective invariant
    x_loc1 = np.vstack([x[i] for i in range(len(x)) if loc[i] == 1])
    x_loc2 = np.vstack([x[i] for i in range(len(x)) if loc[i] == 2])
    assert all(inv1.contains_(x_loc1[i, :].reshape(-1, 1)) for i in range(len(x_loc1))), \
        "Points in loc1 should be in inv1"
    assert all(inv2.contains_(x_loc2[i, :].reshape(-1, 1)) for i in range(len(x_loc2))), \
        "Points in loc2 should be in inv2"
    # time before and after jumps must be the same, but state not
    for i in range(len(t) - 1):
        assert withinTol(t[i][-1], t[i+1][0]), \
            "Time should be continuous across jumps"
        assert not compareMatrices(x[i][-1, :].reshape(-1, 1), x[i+1][0, :].reshape(-1, 1)), \
            "State should change across jumps"

