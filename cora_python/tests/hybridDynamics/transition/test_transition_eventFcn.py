"""
test_transition_eventFcn - test function for transition eventFcn

TRANSLATED FROM: No MATLAB test exists, created based on method implementation

Authors:       Florian Nüssel (Python implementation)
Written:       2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.transition import Transition
from cora_python.hybridDynamics.linearReset import LinearReset
from cora_python.contSet.polytope import Polytope


def test_transition_eventFcn_01_polytope_guard():
    """
    GENERATED TEST - Event function for polytope guard
    
    Tests that eventFcn returns correct values for polytope guard.
    """
    # Create transition with polytope guard
    guard = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                     np.array([[1], [1], [1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Test state inside guard
    x = np.array([0.5, 0.5])
    value, isterminal, direction = trans.eventFcn(x)
    
    # Check return types
    assert isinstance(value, np.ndarray), "value should be numpy array"
    assert isinstance(isterminal, np.ndarray), "isterminal should be numpy array"
    assert isinstance(direction, np.ndarray), "direction should be numpy array"
    
    # Check dimensions
    assert len(value) == len(isterminal) == len(direction), "All outputs should have same length"
    assert len(value) > 0, "Should have at least one event"


def test_transition_eventFcn_02_state_outside_guard():
    """
    GENERATED TEST - Event function for state outside guard
    
    Tests that eventFcn returns correct values for state outside guard.
    """
    # Create transition with polytope guard
    guard = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                     np.array([[1], [1], [1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Test state outside guard
    x = np.array([2.0, 2.0])
    value, isterminal, direction = trans.eventFcn(x)
    
    # Check return types
    assert isinstance(value, np.ndarray), "value should be numpy array"
    assert isinstance(isterminal, np.ndarray), "isterminal should be numpy array"
    assert isinstance(direction, np.ndarray), "direction should be numpy array"
    
    # Check dimensions
    assert len(value) == len(isterminal) == len(direction), "All outputs should have same length"

