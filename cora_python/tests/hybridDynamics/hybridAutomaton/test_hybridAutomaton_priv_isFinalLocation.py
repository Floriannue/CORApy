"""
test_hybridAutomaton_priv_isFinalLocation - test function for hybridAutomaton priv_isFinalLocation

TRANSLATED FROM: No MATLAB test exists, created based on method implementation

Authors:       Florian Nüssel (Python implementation)
Written:       2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.hybridAutomaton import HybridAutomaton
from cora_python.hybridDynamics.location import Location
from cora_python.hybridDynamics.transition import Transition
from cora_python.hybridDynamics.linearReset import LinearReset
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.fullspace import Fullspace
from cora_python.contDynamics.linearSys import LinearSys


def test_hybridAutomaton_priv_isFinalLocation_01_final_location():
    """
    GENERATED TEST - Check if location is final
    
    Tests that priv_isFinalLocation returns True for locations in finalLoc list.
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create location with no transitions (final location) (correct order: name, inv, trans, sys)
    inv = Fullspace(2)
    loc = Location('test', inv, [], sys)
    
    # Create hybrid automaton
    HA = HybridAutomaton('test', [loc])
    
    # Check if location 0 is final (0-based)
    # priv_isFinalLocation is a function, not a method
    from cora_python.hybridDynamics.hybridAutomaton.priv_isFinalLocation import priv_isFinalLocation
    finalLoc = [0]  # 0-based
    result = priv_isFinalLocation(0, finalLoc)
    
    assert result == True, "Location 0 should be final"


def test_hybridAutomaton_priv_isFinalLocation_02_not_final():
    """
    GENERATED TEST - Check if location is not final
    
    Tests that priv_isFinalLocation returns False for locations not in finalLoc list.
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create location with transitions (not final)
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    inv = Fullspace(2)
    loc = Location('test', inv, [trans], sys)
    
    # Create hybrid automaton
    HA = HybridAutomaton('test', [loc])
    
    # Check if location 0 is final (0-based)
    # priv_isFinalLocation is a function, not a method
    from cora_python.hybridDynamics.hybridAutomaton.priv_isFinalLocation import priv_isFinalLocation
    finalLoc = [1]  # 0-based, location 0 is not in list
    result = priv_isFinalLocation(0, finalLoc)
    
    assert result == False, "Location 0 should not be final"


def test_hybridAutomaton_priv_isFinalLocation_03_multiple_final():
    """
    GENERATED TEST - Multiple final locations
    
    Tests that priv_isFinalLocation works with multiple final locations.
    """
    # Create linear systems
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys1 = LinearSys('test1', A, B, c)
    sys2 = LinearSys('test2', A, B, c)
    
    inv = Fullspace(2)
    loc1 = Location('test1', inv, [], sys1)
    loc2 = Location('test2', inv, [], sys2)
    
    # Create hybrid automaton
    HA = HybridAutomaton('test', [loc1, loc2])
    
    # Check if location 1 is final (0-based)
    # priv_isFinalLocation is a function, not a method
    from cora_python.hybridDynamics.hybridAutomaton.priv_isFinalLocation import priv_isFinalLocation
    finalLoc = [0, 1]  # 0-based, both locations are final
    result = priv_isFinalLocation(1, finalLoc)
    
    assert result == True, "Location 1 should be final"

