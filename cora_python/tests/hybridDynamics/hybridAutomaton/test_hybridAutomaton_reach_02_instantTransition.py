"""
test_hybridAutomaton_reach_02_instantTransition - test for reachability
   of hybrid dynamics, where the hybrid automaton contains instant
   transitions (no elapsed time in between two subsequent transitions)

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_reach_02_instantTransition.m

Authors:       Mark Wetzlinger
Written:       17-June-2022
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
from cora_python.g.functions.matlab.validate.check.isequal import isequal


def test_hybridAutomaton_reach_02_instantTransition():
    """
    TRANSLATED TEST - Instant transition reachability test
    
    Tests reachability with instant transitions (no elapsed time between transitions).
    """
    # Parameters
    params = {
        'R0': Zonotope(np.array([[0], [-0.75]]), np.diag([0.05, 0.05])),
        'startLoc': 1,
        'tFinal': 10
    }
    
    # Reachability Options
    options = {
        'timeStep': 0.05,
        'taylorTerms': 1,
        'zonotopeOrder': 5,
        'guardIntersect': 'polytope',
        'enclose': ['box']
    }
    
    # Hybrid Automaton
    # simple construction: 2D automaton with three locations
    # loc1: flow right
    # loc2: flow upwards (irrelevant)
    # loc3: flow left
    # loc4: flow down (irrelevant)
    # however, the transitions loc2->loc3 and loc4->loc1 are instant, thus
    # we should only ever move from right to left and vice versa as the
    # invariants in all locations are the same [-1,1] box
    
    # continuous dynamics
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    c1 = np.array([[1], [0]])
    c2 = np.array([[0], [1]])
    c3 = np.array([[-1], [0]])
    c4 = np.array([[0], [-1]])
    linsys1 = LinearSys('linearSys', A, B, c1)
    linsys2 = LinearSys('linearSys', A, B, c2)
    linsys3 = LinearSys('linearSys', A, B, c3)
    linsys4 = LinearSys('linearSys', A, B, c4)
    
    # invariant set (same for all locations)
    inv = Polytope(Interval(np.array([[-1], [-1]]), np.array([[1], [1]])))
    
    # guard sets
    guard1 = Polytope(np.array([]), np.array([]),
                      np.array([[1, 0]]), np.array([[1]]))
    # fullspace(2) - use large polytope to approximate
    guard2 = Polytope(np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), 
                     np.array([[-100], [100], [-100], [100]]))
    guard3 = Polytope(np.array([]), np.array([]),
                      np.array([[-1, 0]]), np.array([[1]]))
    guard4 = Polytope(np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), 
                     np.array([[-100], [100], [-100], [100]]))
    
    # reset functions
    resetA = np.eye(2)
    resetB = np.zeros((2, 1))
    resetc1 = np.array([[0], [0.15]])
    resetc2 = np.array([[-0.5], [0]])
    resetc3 = np.array([[0], [0.15]])
    resetc4 = np.array([[0.5], [0]])
    reset1 = LinearReset(resetA, resetB, resetc1)
    reset2 = LinearReset(resetA, resetB, resetc2)
    reset3 = LinearReset(resetA, resetB, resetc3)
    reset4 = LinearReset(resetA, resetB, resetc4)
    
    # transitions
    trans1 = Transition(guard1, reset1, 2)
    trans2 = Transition(guard2, reset2, 3)
    trans3 = Transition(guard3, reset3, 4)
    trans4 = Transition(guard4, reset4, 1)
    
    # location objects
    loc1 = Location('right', inv, [trans1], linsys1)
    loc2 = Location('up', inv, [trans2], linsys2)
    loc3 = Location('left', inv, [trans3], linsys3)
    loc4 = Location('down', inv, [trans4], linsys4)
    
    # instantiate hybrid automaton
    HA1 = HybridAutomaton([loc1, loc2, loc3, loc4])
    
    # same hybrid automaton without instant transitions
    # reset functions
    resetc12 = np.array([[-0.5], [0.15]])
    resetc34 = np.array([[0.5], [0.15]])
    reset12 = LinearReset(resetA, resetB, resetc12)
    reset34 = LinearReset(resetA, resetB, resetc34)
    
    # transitions
    trans1_no_inst = Transition(guard1, reset12, 2)
    trans3_no_inst = Transition(guard3, reset34, 1)
    
    # location objects
    loc1_no_inst = Location('right', inv, [trans1_no_inst], linsys1)
    loc2_no_inst = Location('left', inv, [trans3_no_inst], linsys3)
    
    HA2 = HybridAutomaton([loc1_no_inst, loc2_no_inst])
    
    # Reachability Analysis
    R1 = HA1.reach(params, options)
    R2 = HA2.reach(params, options)
    
    # Numerical check
    # all sets need to be equal as well (only check first and last time point)
    i_R1 = 0
    for i in range(len(R2)):
        # skip index if from instant transition
        while len(R1[i_R1].timeInterval.get('set', [])) == 0:
            i_R1 += 1
        
        # same number of sets per branch
        assert len(R1[i_R1].timePoint.get('set', [])) == len(R2[i].timePoint.get('set', [])), \
            f"Branch {i} should have same number of sets"
        
        # same start set and end set
        assert isequal(R1[i_R1].timePoint['set'][0], R2[i].timePoint['set'][0]), \
            f"Branch {i} start sets should be equal"
        assert isequal(R1[i_R1].timePoint['set'][-1], R2[i].timePoint['set'][-1]), \
            f"Branch {i} end sets should be equal"
        
        # increment index of R1
        i_R1 += 1

