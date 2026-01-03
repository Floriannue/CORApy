"""
test_transition_display - test function for display

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/transition/test_transition_display.m

Authors:       Mark Wetzlinger
Written:       19-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contSet.polytope.polytope import Polytope


def test_transition_display_01_empty():
    """
    TRANSLATED TEST - Empty transition display test
    """
    # empty transition
    trans = Transition()
    
    # should not raise error
    trans.display()


def test_transition_display_02_standard():
    """
    TRANSLATED TEST - Standard transition display test
    """
    # init transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]), 
                       np.array([[0], [1]]), np.array([[0], [0]]))
    target = 1
    syncLabel = 'A'
    trans = Transition(guard, reset, target, syncLabel)
    
    # should not raise error
    trans.display()


def test_transition_display_03_array():
    """
    TRANSLATED TEST - Array of transitions display test
    """
    # init transitions
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    target = 1
    
    trans_array = [Transition(), Transition(guard, reset, target)]
    
    # should not raise error
    for trans in trans_array:
        trans.display()

