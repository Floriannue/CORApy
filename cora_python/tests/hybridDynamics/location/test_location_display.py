"""
test_location_display - test function for display

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_display.m

Authors:       Mark Wetzlinger
Written:       19-May-2023
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


def test_location_display_01_empty():
    """
    TRANSLATED TEST - Empty location display test
    """
    # empty location
    loc = Location()
    
    # should not raise error
    loc.display()


def test_location_display_02_standard():
    """
    TRANSLATED TEST - Standard location display test
    """
    # init location
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    dynamics = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                        np.array([[0], [0]]), np.array([[0], [-9.81]]))
    loc = Location(inv, [trans], dynamics)
    
    # should not raise error
    loc.display()


def test_location_display_03_array():
    """
    TRANSLATED TEST - Array of locations display test
    """
    # init locations
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    dynamics = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                        np.array([[0], [0]]), np.array([[0], [-9.81]]))
    
    loc_array = [Location(), Location(inv, [trans], dynamics)]
    
    # should not raise error
    for loc in loc_array:
        loc.display()

