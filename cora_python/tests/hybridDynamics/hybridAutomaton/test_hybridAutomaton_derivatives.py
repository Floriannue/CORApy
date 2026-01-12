"""
test_hybridAutomaton_derivatives - test function for hybridAutomaton derivatives

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


def test_hybridAutomaton_derivatives_01_basic():
    """
    GENERATED TEST - Basic derivatives test
    
    Tests that derivatives method can be called on hybridAutomaton with linear resets.
    Note: derivatives only affects nonlinear resets, so this should pass without error.
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create transition with linear reset
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create location (correct order: name, inv, trans, sys)
    inv = Fullspace(2)
    loc = Location('test', inv, [trans], sys)
    
    # Create hybrid automaton
    HA = HybridAutomaton('test', [loc])
    
    # Call derivatives (should not fail, but won't do anything for linear resets)
    HA_result = HA.derivatives()
    
    # HybridAutomaton should be returned
    assert HA_result is not None, "derivatives should return hybridAutomaton"
    assert len(HA_result.location) == 1, "Should have one location"


def test_hybridAutomaton_derivatives_02_specific_locations():
    """
    GENERATED TEST - Derivatives for specific locations
    
    Tests that derivatives can be called for specific location indices.
    """
    # Create linear systems
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys1 = LinearSys('test1', A, B, c)
    sys2 = LinearSys('test2', A, B, c)
    
    # Create transitions
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    trans1 = Transition(guard, reset, 1)
    trans2 = Transition(guard, reset, 1)  # target must be positive (1-based)
    
    # Create locations (correct order: name, inv, trans, sys)
    inv = Fullspace(2)
    loc1 = Location('test1', inv, [trans1], sys1)
    loc2 = Location('test2', inv, [trans2], sys2)
    
    # Create hybrid automaton
    HA = HybridAutomaton('test', [loc1, loc2])
    
    # Call derivatives for first location only (0-based index)
    HA_result = HA.derivatives(locIdx=[0])
    
    # HybridAutomaton should be returned
    assert HA_result is not None, "derivatives should return hybridAutomaton"
    assert len(HA_result.location) == 2, "Should have two locations"

