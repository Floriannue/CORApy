"""
test_hybridAutomaton_reach_unsafeSet - test function for reachability
   analysis of hybrid systems; here, we ensure that a reachable set that
   has already left the invariant cannot intersect an unsafe set

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_reach_unsafeSet.m

Authors:       Mark Wetzlinger
Written:       21-October-2024
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
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.specification.specification.specification import Specification


def test_hybridAutomaton_reach_unsafeSet():
    """
    TRANSLATED TEST - Unsafe set reachability test
    
    Tests that reachable sets that have left the invariant cannot intersect unsafe sets.
    """
    # simple automaton: the reachable set moves from left to right and vice
    # versa between two vertical guards, behind which there are unsafe sets
    inv = Polytope(Interval(np.array([[-1], [-100]]), np.array([[1], [100]])))
    guard_right = Polytope(np.array([]), np.array([]),
                           np.array([[1, 0]]), np.array([[1]]))
    guard_left = Polytope(np.array([]), np.array([]),
                          np.array([[1, 0]]), np.array([[-1]]))
    reset = LinearReset(np.eye(2), np.zeros((2, 1)), np.array([[0], [1]]))
    trans1 = Transition(guard_right, reset, 2)
    trans2 = Transition(guard_left, reset, 1)
    
    # dynamics
    linsys1 = LinearSys('linearSys', np.zeros((2, 2)), np.zeros((2, 1)), np.array([[1], [0]]))
    linsys2 = LinearSys('linearSys', np.zeros((2, 2)), np.zeros((2, 1)), np.array([[-1], [0]]))
    
    # init locations and hybrid automaton
    loc1 = Location('to-right', inv, [trans1], linsys1)
    loc2 = Location('to-left', inv, [trans2], linsys2)
    HA = HybridAutomaton([loc1, loc2])
    
    # model parameters: make initial set wide enough so that it reaches the
    # unsafe set behind the guard set before fully exiting the invariant set;
    # also, make the time horizon long enough so that there are a couple of
    # bounces left and right
    params = {
        'R0': Zonotope(np.array([[0], [0]]), np.array([[0.75, 0], [0, 0.25]])),
        'startLoc': 1,
        'finalLoc': 0,
        'tFinal': 5
    }
    
    # settings for continuous reachability
    options = {
        'timeStep': 0.05,
        'taylorTerms': 10,
        'zonotopeOrder': 20,
        'guardIntersect': 'polytope',
        'enclose': ['pca']
    }
    
    # specifications: unsafe sets
    S_right = Interval(np.array([[1.5], [-0.1]]), np.array([[2], [0.1]]))
    S_left = Interval(np.array([[-2], [-0.1]]), np.array([[-1.5], [0.1]]))
    spec = [
        Specification(S_right, 'unsafeSet', 1),
        Specification(S_left, 'unsafeSet', 2)
    ]
    
    # compute reachable set
    R = HA.reach(params, options, spec)
    
    # final set must reach time horizon (no abortion due to unsafe sets)
    final_time = R[-1].timePoint['time'][-1]
    if isinstance(final_time, Interval):
        assert final_time.contains_(params['tFinal']), \
            "Final time should reach time horizon"
    else:
        assert final_time >= params['tFinal'] - 1e-6, \
            "Final time should reach time horizon"

