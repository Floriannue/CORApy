"""
test_transition_transition - unit test for constructor of the class transition

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/transition/test_transition_transition.m

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
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_transition_01_empty():
    """
    TRANSLATED TEST - Empty constructor test
    
    Tests Transition with no arguments.
    MATLAB: guard = []; reset = []; target = [];
    """
    trans = Transition()
    
    # MATLAB uses empty arrays [], not None
    assert isinstance(trans.guard, np.ndarray) and trans.guard.size == 0, "guard should be empty array"
    assert isinstance(trans.reset, np.ndarray) and trans.reset.size == 0, "reset should be empty array"
    assert isinstance(trans.target, np.ndarray) and trans.target.size == 0, "target should be empty array"
    assert trans.syncLabel == '', "syncLabel should be empty string"


def test_transition_02_standard_instantiation():
    """
    TRANSLATED TEST - Standard instantiation test
    
    Tests Transition with guard, reset, target, and syncLabel.
    """
    # guard set
    guard_2D = Polytope(np.array([[0, 1]]), np.array([[0]]), 
                       np.array([[1, 0]]), np.array([[0]]))
    
    # reset function (linear)
    reset_2D = LinearReset(np.array([[0, 0],
                                     [0, 0.2]]),
                          np.array([[1], [0]]),
                          np.array([[0], [-1]]))
    
    # target location
    target = 2
    
    # synchronization label
    syncLabel = 'on'
    
    # check standard instantiation
    trans = Transition(guard_2D, reset_2D, target, syncLabel)
    
    # Check guard
    assert trans.guard is not None, "guard should be set"
    # Note: isequal for polytopes might need special handling
    
    # Check reset
    assert trans.reset is not None, "reset should be set"
    np.testing.assert_allclose(trans.reset.A, reset_2D.A, 
                               err_msg="reset.A should match")
    np.testing.assert_allclose(trans.reset.c, reset_2D.c, 
                               err_msg="reset.c should match")
    assert trans.reset.preStateDim == 2, "reset.preStateDim should be 2"
    assert trans.reset.inputDim == 1, "reset.inputDim should be 1"
    
    # Check target
    assert trans.target == target, "target should match"
    
    # Check syncLabel
    assert trans.syncLabel == syncLabel, "syncLabel should match"


def test_transition_03_wrong_instantiations():
    """
    TRANSLATED TEST - Wrong instantiations test
    
    Tests that wrong instantiations raise appropriate errors.
    """
    guard_2D = Polytope(np.array([[0, 1]]), np.array([[0]]), 
                       np.array([[1, 0]]), np.array([[0]]))
    reset_2D = LinearReset(np.array([[0, 0],
                                     [0, 0.2]]),
                          np.array([[1], [0]]),
                          np.array([[0], [-1]]))
    target = 2
    syncLabel = 'on'
    
    # one input argument, but not copy constructor
    with pytest.raises(Exception):  # CORA:wrongValue
        Transition(guard_2D)
    
    # not enough input arguments
    with pytest.raises(Exception):  # CORA:numInputArgsConstructor
        Transition(guard_2D, reset_2D)
    
    # too many input arguments
    with pytest.raises(Exception):  # CORA:numInputArgsConstructor
        Transition(guard_2D, reset_2D, target, syncLabel, syncLabel)
    
    # guard set is defined using a wrong set representation
    wrong_guard = Zonotope(np.array([[2], [1]]), np.eye(2))
    with pytest.raises(Exception):  # CORA:wrongValue
        Transition(wrong_guard, reset_2D, target)
    
    # target not an integer
    with pytest.raises(Exception):  # CORA:wrongValue
        Transition(guard_2D, reset_2D, 1.5)
    
    # target negative
    with pytest.raises(Exception):  # CORA:wrongValue
        Transition(guard_2D, reset_2D, -1)
    
    # sync label is numeric
    with pytest.raises(Exception):  # CORA:wrongValue
        Transition(guard_2D, reset_2D, target, 1)
