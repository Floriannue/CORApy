"""
test_location_derivatives - test function for location derivatives

TRANSLATED FROM: No MATLAB test exists, created based on method implementation

Authors:       Florian Nüssel (Python implementation)
Written:       2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.location import Location
from cora_python.hybridDynamics.transition import Transition
from cora_python.hybridDynamics.linearReset import LinearReset
from cora_python.contSet.polytope import Polytope
from cora_python.contDynamics.linearSys import LinearSys


def test_location_derivatives_01_basic():
    """
    GENERATED TEST - Basic derivatives test
    
    Tests that derivatives method can be called on location with linear resets.
    Note: derivatives only affects nonlinear resets, so this should pass without error.
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create transition with linear reset (derivatives won't affect it)
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create location (correct order: name, inv, trans, sys)
    inv = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    loc = Location('test', inv, [trans], sys)
    
    # Call derivatives (should not fail, but won't do anything for linear resets)
    loc_result = loc.derivatives()
    
    # Location should be returned
    assert loc_result is not None, "derivatives should return location"
    assert len(loc_result.transition) == 1, "Should have one transition"
    assert isinstance(loc_result.transition[0].reset, LinearReset), "Reset should remain linear"


def test_location_derivatives_02_specific_transitions():
    """
    GENERATED TEST - Derivatives for specific transitions
    
    Tests that derivatives can be called for specific transition indices.
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create transitions
    guard1 = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    guard2 = Polytope(np.array([[0, 1], [0, -1]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    trans1 = Transition(guard1, reset, 1)
    trans2 = Transition(guard2, reset, 2)
    
    # Create location (correct order: name, inv, trans, sys)
    inv = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    loc = Location('test', inv, [trans1, trans2], sys)
    
    # Call derivatives for first transition only (0-based index)
    loc_result = loc.derivatives(transIdx=[0])
    
    # Location should be returned
    assert loc_result is not None, "derivatives should return location"
    assert len(loc_result.transition) == 2, "Should have two transitions"

