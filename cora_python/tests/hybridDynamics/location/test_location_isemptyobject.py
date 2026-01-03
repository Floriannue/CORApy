"""
test_location_isemptyobject - test function for isemptyobject

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/location/test_location_isemptyobject.m

Authors:       Mark Wetzlinger
Written:       16-May-2023
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


def test_location_isemptyobject_01_empty():
    """
    TRANSLATED TEST - Empty location test
    """
    # empty location
    assert Location().isemptyobject(), "Empty location should be empty"


def test_location_isemptyobject_02_nonempty():
    """
    TRANSLATED TEST - Non-empty location test
    """
    # non-empty location
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    dynamics = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                         np.array([[0], [0]]), np.array([[0], [-9.81]]))
    
    assert not Location(inv, [trans], dynamics).isemptyobject(), \
        "Non-empty location should not be empty"
    
    # array of locations
    loc_array = [Location(), Location(inv, [trans], dynamics)]
    empty_results = [loc.isemptyobject() for loc in loc_array]
    assert empty_results == [True, False], \
        "Array of locations should have correct empty results"

