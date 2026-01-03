"""
test_hybridAutomaton_display - test function for display

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_display.m

Authors:       Mark Wetzlinger
Written:       08-May-2023
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


def test_hybridAutomaton_display_01_empty():
    """
    TRANSLATED TEST - Empty hybridAutomaton display test
    """
    # empty hybridAutomaton
    HA = HybridAutomaton()
    
    # should not raise error
    HA.display()


def test_hybridAutomaton_display_02_standard():
    """
    TRANSLATED TEST - Standard hybridAutomaton display test
    """
    # init hybridAutomaton
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    dynamics = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                        np.array([[0], [0]]), np.array([[0], [-9.81]]))
    loc = Location(inv, [trans], dynamics)
    HA = HybridAutomaton([loc])
    
    # should not raise error
    HA.display()


def test_hybridAutomaton_display_03_array():
    """
    TRANSLATED TEST - Array of hybridAutomata display test
    """
    # init hybridAutomata
    inv = Polytope(np.array([[-1, 0]]), np.array([[0]]))
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    dynamics = LinearSys('linearSys', np.array([[0, 1], [0, 0]]), 
                        np.array([[0], [0]]), np.array([[0], [-9.81]]))
    loc = Location(inv, [trans], dynamics)
    
    HA_array = [HybridAutomaton(), HybridAutomaton([loc])]
    
    # should not raise error
    for HA in HA_array:
        HA.display()

