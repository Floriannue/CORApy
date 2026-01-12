"""
test_location_eventFcn - test function for location eventFcn

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
from cora_python.contSet.fullspace import Fullspace
from cora_python.contDynamics.linearSys import LinearSys


def test_location_eventFcn_01_basic():
    """
    GENERATED TEST - Basic eventFcn test
    
    Tests that eventFcn returns a callable function.
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create transition
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create location with fullspace invariant (correct order: name, inv, trans, sys)
    inv = Fullspace(2)
    loc = Location('test', inv, [trans], sys)
    
    # Get event function
    event_func = loc.eventFcn()
    
    # Should be callable
    assert callable(event_func), "eventFcn should return a callable function"
    
    # Test calling it
    t = 0.0
    x = np.array([0.5, 0.5])
    value, isterminal, direction = event_func(t, x)
    
    # Check return types
    assert isinstance(value, np.ndarray), "value should be numpy array"
    assert isinstance(isterminal, np.ndarray), "isterminal should be numpy array"
    assert isinstance(direction, np.ndarray), "direction should be numpy array"
    
    # Check dimensions
    assert len(value) == len(isterminal) == len(direction), "All outputs should have same length"


def test_location_eventFcn_02_empty_invariant():
    """
    GENERATED TEST - Event function with empty invariant
    
    Tests that eventFcn works with empty invariant (emptySet).
    """
    # Create linear system
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    c = np.array([[0], [0]])
    sys = LinearSys('test', A, B, c)
    
    # Create transition
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create location with empty invariant (correct order: name, inv, trans, sys)
    from cora_python.contSet.emptySet import EmptySet
    inv = EmptySet(2)
    loc = Location('test', inv, [trans], sys)
    
    # Get event function
    event_func = loc.eventFcn()
    
    # Should be callable
    assert callable(event_func), "eventFcn should return a callable function"
    
    # Test calling it
    t = 0.0
    x = np.array([0.5, 0.5])
    value, isterminal, direction = event_func(t, x)
    
    # Should only have guard events (no invariant events)
    assert isinstance(value, np.ndarray), "value should be numpy array"
    assert len(value) > 0, "Should have guard events"

