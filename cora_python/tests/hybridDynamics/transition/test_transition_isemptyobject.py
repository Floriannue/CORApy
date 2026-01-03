"""
test_transition_isemptyobject - unit test for emptiness check

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/transition/test_transition_isemptyobject.m

Authors:       Mark Wetzlinger
Written:       15-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contSet.polytope.polytope import Polytope


def test_transition_isemptyobject_01_empty():
    """
    TRANSLATED TEST - Empty transition test
    """
    # empty object
    assert Transition().isemptyobject(), "Empty transition should be empty"


def test_transition_isemptyobject_02_nonempty():
    """
    TRANSLATED TEST - Non-empty transition test
    """
    # transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[0, 0], [0, 0.2]]))
    target = 2
    syncLabel = 'on'
    
    trans = Transition(guard, reset, target, syncLabel)
    assert not trans.isemptyobject(), "Non-empty transition should not be empty"
    
    trans = Transition(guard, reset, target)
    assert not trans.isemptyobject(), "Non-empty transition should not be empty"
    
    # array of transitions
    trans_array = [Transition(), Transition(guard, reset, target)]
    empty_results = [t.isemptyobject() for t in trans_array]
    assert empty_results == [True, False], \
        "Array of transitions should have correct empty results"

