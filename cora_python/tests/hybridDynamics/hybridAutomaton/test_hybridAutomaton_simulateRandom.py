"""
test_hybridAutomaton_simulateRandom - test function for simulateRandom

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_simulateRandom.m

Authors:       Mark Wetzlinger
Written:       16-May-2023
Last update:   15-October-2024 (MW, move to test_)
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
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_hybridAutomaton_simulateRandom_01_basic():
    """
    TRANSLATED TEST - Basic simulateRandom test
    
    Tests random simulation of hybrid automaton trajectories.
    """
    # continuous dynamics
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [0]])
    c = np.array([[0], [-9.81]])
    linsys = LinearSys('linearSys', A, B, c)
    
    # rebound factor
    alpha = -0.75
    
    # invariant set
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    
    # guard sets
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[1, 0]]), np.array([[0]]))
    
    # reset function
    reset = LinearReset(np.array([[0, 0], [0, alpha]]), np.zeros((2, 1)), np.zeros((2, 1)))
    
    # self-transition
    trans = Transition(guard, reset, 1)
    
    # location object
    loc = Location('always', inv, [trans], linsys)
    
    # hybrid automata
    HA = HybridAutomaton([loc])
    
    # model parameters
    params = {
        'R0': Zonotope(np.array([[0.3], [0]]), np.diag([0.025, 0.025])),
        'startLoc': 1,
        'finalLoc': 0,
        'tFinal': 0.5
    }
    
    # settings for continuous reachability
    options = {
        'timeStep': 0.05,
        'taylorTerms': 10,
        'zonotopeOrder': 20,
        'guardIntersect': 'polytope',
        'enclose': ['pca']
    }
    
    # compute reachable set
    R = HA.reach(params, options)
    
    # simulate random trajectories
    simOpt = {
        'points': 5
    }
    simRes = HA.simulateRandom(params, simOpt)
    
    # five trajectories
    assert len(simRes) == simOpt['points'], "Should have 5 trajectories"
    
    # correct start and end time
    assert all(traj['t'][0][0] == 0 and traj['t'][-1][-1] == params['tFinal'] 
               for traj in simRes), \
        "Each trajectory should start at 0 and end at tFinal"
    
    # correct start point for each trajectory
    from cora_python.contSet.zonotope.contains_ import contains_
    assert all(contains_(params['R0'], traj['x'][0][0, :].reshape(-1, 1), 'exact', 
                        np.finfo(float).eps, 0, False, False) 
               for traj in simRes), \
        "Start points should be in R0"
    
    # correct start location for each trajectory
    assert all(traj['loc'][0] == params['startLoc'] for traj in simRes), \
        "Start location should match"
    
    # time is strictly increasing
    for traj in simRes:
        all_times = np.concatenate(traj['t'])
        assert np.all(np.diff(all_times) > -np.finfo(float).eps), \
            "Time should be strictly increasing"
    
    # check if simulations are contained in reachable set
    from cora_python.g.classes.reachSet.contains_ import contains_
    assert contains_(R, simRes), "Simulations should be contained in reachable set"
    
    # simulate edge case with a single input segment
    simOpt['nrConstInp'] = 1
    HA.simulateRandom(params, simOpt)  # Should not raise error

