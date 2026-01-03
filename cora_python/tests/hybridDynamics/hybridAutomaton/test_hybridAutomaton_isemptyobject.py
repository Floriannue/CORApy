"""
test_hybridAutomaton_isemptyobject - test function for isemptyobject

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_isemptyobject.m

Authors:       Mark Wetzlinger
Written:       15-May-2023
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


def test_hybridAutomaton_isemptyobject_01_empty():
    """
    TRANSLATED TEST - Empty hybridAutomaton test
    """
    # empty object
    assert HybridAutomaton().isemptyobject(), "Empty hybridAutomaton should be empty"


def test_hybridAutomaton_isemptyobject_02_nonempty():
    """
    TRANSLATED TEST - Non-empty hybridAutomaton test
    """
    # hybridAutomaton
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    dynamics = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                        np.array([[0], [0]]), np.array([[0], [-9.81]]))
    loc = Location(inv, [trans], dynamics)
    
    HA = HybridAutomaton([loc])
    assert not HA.isemptyobject(), "Non-empty hybridAutomaton should not be empty"
    
    # array of hybridAutomata
    HA_array = [HybridAutomaton(), HybridAutomaton([loc])]
    empty_results = [ha.isemptyobject() for ha in HA_array]
    assert empty_results == [True, False], \
        "Array of hybridAutomata should have correct empty results"

