"""
test_transition_isequal - test function for isequal

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/transition/test_transition_isequal.m

Authors:       Mark Wetzlinger
Written:       26-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contSet.polytope.polytope import Polytope


def test_transition_isequal_01_empty():
    """
    TRANSLATED TEST - Empty transition equality test
    """
    # two empty transitions
    assert Transition().isequal(Transition()), "Empty transitions should be equal"


def test_transition_isequal_02_same_transition():
    """
    TRANSLATED TEST - Same transition test
    """
    # guard set
    guard1 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 0]]), np.array([[0]]))
    
    # reset function
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.array([[0], [1]]), np.array([[0], [0]]))
    
    # target location
    target1 = 1
    
    # synchronization label
    syncLabel1 = 'A'
    
    # same transition
    assert Transition(guard1, reset1, target1, syncLabel1).isequal(
        Transition(guard1, reset1, target1, syncLabel1)), \
        "Same transitions should be equal"
    
    # same transitions in array
    guard2 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 1]]), np.array([[0]]))
    reset2 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                        np.array([[0], [1]]), np.array([[1], [0]]))
    target2 = 2
    syncLabel2 = 'B'
    
    trans_array1 = [Transition(guard1, reset1, target1, syncLabel1),
                    Transition(guard2, reset2, target2, syncLabel2)]
    trans_array2 = [Transition(guard1, reset1, target1, syncLabel1),
                    Transition(guard2, reset2, target2, syncLabel2)]
    
    # Note: isequal for arrays might need special handling
    # For now, check individual elements
    assert all(t1.isequal(t2) for t1, t2 in zip(trans_array1, trans_array2)), \
        "Arrays of same transitions should be equal"


def test_transition_isequal_03_different_guard():
    """
    TRANSLATED TEST - Different guard test
    """
    # guard set
    guard1 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 0]]), np.array([[0]]))
    guard2 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 1]]), np.array([[0]]))
    
    # reset function
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.array([[0], [1]]), np.array([[0], [0]]))
    
    # target location
    target1 = 1
    
    # synchronization label
    syncLabel1 = 'A'
    
    # different guard set
    assert not Transition(guard1, reset1, target1, syncLabel1).isequal(
        Transition(guard2, reset1, target1, syncLabel1)), \
        "Transitions with different guards should not be equal"


def test_transition_isequal_04_different_reset():
    """
    TRANSLATED TEST - Different reset test
    """
    # guard set
    guard1 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 0]]), np.array([[0]]))
    
    # reset function
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.array([[0], [1]]), np.array([[0], [0]]))
    reset2 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.array([[0], [1]]), np.array([[1], [0]]))
    
    # target location
    target1 = 1
    
    # synchronization label
    syncLabel1 = 'A'
    
    # different reset function
    assert not Transition(guard1, reset1, target1, syncLabel1).isequal(
        Transition(guard1, reset2, target1, syncLabel1)), \
        "Transitions with different resets should not be equal"


def test_transition_isequal_05_different_target():
    """
    TRANSLATED TEST - Different target test
    """
    # guard set
    guard1 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 0]]), np.array([[0]]))
    
    # reset function
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.array([[0], [1]]), np.array([[0], [0]]))
    
    # target location
    target1 = 1
    target2 = 2
    
    # synchronization label
    syncLabel1 = 'A'
    
    # different target location
    assert not Transition(guard1, reset1, target1, syncLabel1).isequal(
        Transition(guard1, reset1, target2, syncLabel1)), \
        "Transitions with different targets should not be equal"


def test_transition_isequal_06_different_syncLabel():
    """
    TRANSLATED TEST - Different synchronization label test
    """
    # guard set
    guard1 = Polytope(np.array([[0, 1]]), np.array([[0]]),
                      np.array([[-1, 0]]), np.array([[0]]))
    
    # reset function
    reset1 = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                         np.array([[0], [1]]), np.array([[0], [0]]))
    
    # target location
    target1 = 1
    
    # synchronization label
    syncLabel1 = 'A'
    syncLabel2 = 'B'
    
    # different synchronization label
    assert not Transition(guard1, reset1, target1, syncLabel1).isequal(
        Transition(guard1, reset1, target1, syncLabel2)), \
        "Transitions with different sync labels should not be equal"

