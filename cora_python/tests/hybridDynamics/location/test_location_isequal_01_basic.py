"""
test_location_isequal - test function for isequal

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_isequal.m

Authors:       Mark Wetzlinger
Written:       26-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location.location import Location
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.polytope.polytope import Polytope


def test_location_isequal_01_empty():
    """
    TRANSLATED TEST - Empty location equality test
    """
    # empty locations
    assert Location().isequal(Location()), "Empty locations should be equal"


def test_location_isequal_02_same_location():
    """
    TRANSLATED TEST - Same location test
    """
    # invariant
    inv1 = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    
    # reset function
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    
    # transition
    trans1 = Transition(guard, reset1, 1)
    
    # flow equation
    dynamics1 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.81]]))
    
    # same location
    assert Location(inv1, [trans1], dynamics1).isequal(
        Location(inv1, [trans1], dynamics1)), "Same locations should be equal"
    
    # same location with transitions in different order
    trans2 = Transition(guard, reset1, 2)
    assert Location(inv1, [trans1, trans2], dynamics1).isequal(
        Location(inv1, [trans2, trans1], dynamics1)), \
        "Locations with same transitions in different order should be equal"


def test_location_isequal_03_different_invariant():
    """
    TRANSLATED TEST - Different invariant test
    """
    # invariant
    inv1 = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    inv2 = Polytope(np.array([[-1, 0]]), np.array([[0.5]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans1 = Transition(guard, reset1, 1)
    
    # flow equation
    dynamics1 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.81]]))
    
    # different invariant
    assert not Location(inv1, [trans1], dynamics1).isequal(
        Location(inv2, [trans1], dynamics1)), \
        "Locations with different invariants should not be equal"


def test_location_isequal_04_different_dynamics():
    """
    TRANSLATED TEST - Different dynamics test
    """
    # invariant
    inv1 = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans1 = Transition(guard, reset1, 1)
    
    # flow equation
    dynamics1 = LinearSys('linearSys1', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.81]]))
    dynamics2 = LinearSys('linearSys2', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.80]]))
    
    # different dynamics
    assert not Location(inv1, [trans1], dynamics1).isequal(
        Location(inv1, [trans1], dynamics2)), \
        "Locations with different dynamics should not be equal"


def test_location_isequal_05_different_transition():
    """
    TRANSLATED TEST - Different transition test
    """
    # invariant
    inv1 = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans1 = Transition(guard, reset1, 1)
    trans2 = Transition(guard, reset1, 2)
    
    # flow equation
    dynamics1 = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                          np.array([[0], [0]]), np.array([[0], [-9.81]]))
    
    # different transition
    assert not Location(inv1, [trans1], dynamics1).isequal(
        Location(inv1, [trans2], dynamics1)), \
        "Locations with different transitions should not be equal"
    
    # different transition (different reset)
    reset2 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.zeros((2, 1)), np.array([[1], [0]]))
    trans3 = Transition(guard, reset2, 2)
    assert not Location(inv1, [trans1], dynamics1).isequal(
        Location(inv1, [trans3], dynamics1)), \
        "Locations with different transitions should not be equal"
    
    # different number of transitions
    assert not Location(inv1, [trans1], dynamics1).isequal(
        Location(inv1, [trans1, trans2], dynamics1)), \
        "Locations with different number of transitions should not be equal"

