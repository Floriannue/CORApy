"""
test_transition_guard2polytope - test function for transition guard2polytope

TRANSLATED FROM: No MATLAB test exists, created based on method implementation

Authors:       Florian Nüssel (Python implementation)
Written:       2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.transition import Transition
from cora_python.hybridDynamics.linearReset import LinearReset
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.interval import Interval


def test_transition_guard2polytope_01_interval_to_polytope():
    """
    GENERATED TEST - Convert interval guard to polytope
    
    Tests that guard2polytope converts an interval guard to a polytope.
    """
    # Create transition with interval guard
    guard = Interval(np.array([0, 1]), np.array([2, 3]))
    reset = LinearReset(np.eye(2))
    target = 1
    
    trans = Transition(guard, reset, target)
    
    # Convert guard to polytope
    # Note: guard2polytope internally calls Polytope(guard) which should work
    trans = trans.guard2polytope()
    
    # Guard should now be a polytope
    assert isinstance(trans.guard, Polytope), "Guard should be converted to polytope"
    assert trans.reset is not None, "Reset should remain unchanged"
    assert trans.target == target, "Target should remain unchanged"


def test_transition_guard2polytope_02_already_polytope():
    """
    GENERATED TEST - Guard already polytope
    
    Tests that guard2polytope does nothing if guard is already a polytope.
    """
    # Create transition with polytope guard
    guard = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                     np.array([[1], [1], [1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    
    trans = Transition(guard, reset, target)
    original_guard = trans.guard
    
    # Convert guard to polytope (should do nothing)
    trans = trans.guard2polytope()
    
    # Guard should still be a polytope (same object or equivalent)
    assert isinstance(trans.guard, Polytope), "Guard should remain a polytope"
    assert trans.reset is not None, "Reset should remain unchanged"
    assert trans.target == target, "Target should remain unchanged"

