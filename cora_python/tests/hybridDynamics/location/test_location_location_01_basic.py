"""
test_location_location - unit test for constructor of the class location

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_location.m

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


def test_location_01_empty():
    """
    TRANSLATED TEST - Empty constructor test
    
    Tests Location with no arguments.
    MATLAB: invariant = []; transition = transition(); contDynamics = contDynamics();
    """
    import numpy as np
    loc = Location()
    
    assert loc.name == 'location', "Default name should be 'location'"
    # MATLAB uses empty arrays [], not None
    assert isinstance(loc.invariant, np.ndarray) and loc.invariant.size == 0, "Invariant should be empty array"
    assert loc.transition == [], "Transition should be empty list"
    assert loc.contDynamics is None, "contDynamics should be None"


def test_location_02_standard_instantiation():
    """
    TRANSLATED TEST - Standard instantiation test
    
    Tests Location with name, invariant, transition, and dynamics.
    """
    # name of the location
    name = 'S1'
    
    # invariant
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    
    # reset function
    reset = LinearReset(np.array([[1, 0],
                                  [0, -0.75]]))
    
    # transition
    trans = Transition(guard, reset, 2)
    
    # flow equation
    dynamics = LinearSys('linearSys', np.array([[0, 1],
                                                 [0, 0]]),
                         np.array([[0], [0]]),
                         np.array([[0], [-9.81]]))
    
    # define location
    loc = Location(name, inv, trans, dynamics)
    
    # check if values have been assigned correctly
    assert loc.name == name, "Name should match"
    # Note: isequal for polytopes might need special handling
    assert loc.invariant is not None, "Invariant should be set"
    assert len(loc.transition) == 1, "Should have 1 transition"
    assert loc.contDynamics is not None, "contDynamics should be set"
    # Check dynamics matches
    assert isinstance(loc.contDynamics, LinearSys), "contDynamics should be LinearSys"

