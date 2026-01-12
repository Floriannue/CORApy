"""
test_transition_convGuard - test function for transition convGuard

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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_transition_convGuard_01_polytope():
    """
    GENERATED TEST - Convert guard to polytope and intersect with invariant
    
    Tests that convGuard converts guard to polytope and intersects with invariant.
    """
    # Import Interval for type checking
    from cora_python.contSet.interval.interval import Interval
    
    # Create transition with interval guard (needs conversion to polytope)
    guard = Interval(np.array([0, 1]), np.array([2, 3]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Verify guard is initially an Interval
    assert isinstance(trans.guard, Interval), "Guard should start as Interval"
    
    # Create invariant (same dimension, compatible)
    inv = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                   np.array([[1.5], [1.5], [2.5], [2.5]]))
    
    # Options
    options = {'guardIntersect': 'polytope'}
    
    # Convert guard (should convert Interval to Polytope and intersect with invariant)
    trans = trans.convGuard(inv, options)
    
    # Guard should now be a polytope (converted from Interval)
    assert isinstance(trans.guard, Polytope), "Guard should be converted to polytope"
    assert trans.reset is not None, "Reset should remain unchanged"
    assert trans.target == target, "Target should remain unchanged"


def test_transition_convGuard_01b_already_polytope():
    """
    GENERATED TEST - Guard already polytope, just intersect
    
    Tests that convGuard works when guard is already a polytope.
    """
    # Create transition with polytope guard (already a polytope, so conversion is skipped)
    guard = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                     np.array([[2], [0], [3], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create invariant (same dimension, compatible)
    inv = Polytope(np.array([[1, 0], [-1, 0]]), 
                   np.array([[1.5], [1.5]]))
    
    # Options
    options = {'guardIntersect': 'polytope'}
    
    # Convert guard (should intersect with invariant)
    trans = trans.convGuard(inv, options)
    
    # Guard should be a polytope
    assert isinstance(trans.guard, Polytope), "Guard should be a polytope"
    assert trans.reset is not None, "Reset should remain unchanged"
    assert trans.target == target, "Target should remain unchanged"


def test_transition_convGuard_02_conZonotope():
    """
    GENERATED TEST - Convert guard to conZonotope
    
    Tests that convGuard converts polytope guard to conZonotope.
    """
    # Create transition with polytope guard
    guard = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), 
                     np.array([[1], [1], [1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create invariant (not used for conZonotope)
    inv = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    
    # Options
    options = {'guardIntersect': 'conZonotope'}
    
    # Convert guard
    trans = trans.convGuard(inv, options)
    
    # Guard should be a conZonotope
    from cora_python.contSet.conZonotope import ConZonotope
    assert isinstance(trans.guard, ConZonotope), "Guard should be a conZonotope"
    assert trans.reset is not None, "Reset should remain unchanged"
    assert trans.target == target, "Target should remain unchanged"


def test_transition_convGuard_03_invalid_method():
    """
    GENERATED TEST - Invalid guard intersection method
    
    Tests that convGuard raises error for invalid method.
    """
    # Create transition
    guard = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    reset = LinearReset(np.eye(2))
    target = 1
    trans = Transition(guard, reset, target)
    
    # Create invariant
    inv = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[1], [1]]))
    
    # Options with invalid method
    options = {'guardIntersect': 'invalid_method'}
    
    # Should raise error
    with pytest.raises(CORAerror):
        trans.convGuard(inv, options)

