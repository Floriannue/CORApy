"""
test_hybridAutomaton_isequal - test function for isequal

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_isequal.m

Authors:       Mark Wetzlinger
Written:       26-November-2022
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


def test_hybridAutomaton_isequal_01_empty():
    """
    TRANSLATED TEST - Empty hybridAutomaton equality test
    """
    # two empty hybridAutomata
    assert HybridAutomaton().isequal(HybridAutomaton()), "Empty hybridAutomata should be equal"


def test_hybridAutomaton_isequal_02_same():
    """
    TRANSLATED TEST - Same hybridAutomaton test
    """
    # init locations
    inv1 = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans1 = Transition(guard, reset1, 2)
    dynamics1 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.81]]))
    loc1 = Location(inv1, [trans1], dynamics1)
    
    inv2 = Polytope(np.array([[-1, 0]]), np.array([[0.5]]))
    reset2 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans2 = Transition(guard, reset2, 1)
    dynamics2 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.80]]))
    loc2 = Location(inv2, [trans2], dynamics2)
    
    # same hybridAutomaton
    HA1 = HybridAutomaton([loc1, loc2])
    HA2 = HybridAutomaton([loc1, loc2])
    
    assert HA1.isequal(HA2), "Same hybridAutomata should be equal"


def test_hybridAutomaton_isequal_03_different_locations():
    """
    TRANSLATED TEST - Different locations test
    """
    # init locations
    inv1 = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans1 = Transition(guard, reset1, 2)  # Points to location 2
    dynamics1 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.81]]))
    loc1 = Location(inv1, [trans1], dynamics1)
    
    # Create a second location for loc1 to point to (target = 2)
    inv1b = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    trans1b = Transition(guard, reset1, 1)  # Points back to location 1
    loc1b = Location(inv1b, [trans1b], dynamics1)
    
    inv2 = Polytope(np.array([[-1, 0]]), np.array([[0.5]]))
    reset2 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans2 = Transition(guard, reset2, 1)  # Points to location 1
    dynamics2 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.80]]))
    loc2 = Location(inv2, [trans2], dynamics2)
    
    # Create a second location for loc2 to point to (target = 1, but we need 2 locations)
    inv2b = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    trans2b = Transition(guard, reset2, 1)  # Points to location 1
    loc2b = Location(inv2b, [trans2b], dynamics2)
    
    # different locations - both automata have 2 locations
    HA1 = HybridAutomaton([loc1, loc1b])
    HA2 = HybridAutomaton([loc2, loc2b])
    
    assert not HA1.isequal(HA2), "HybridAutomata with different locations should not be equal"

