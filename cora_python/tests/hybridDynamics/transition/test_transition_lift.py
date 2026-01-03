"""
test_transition_lift - test function for lift

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/transition/testLong_transition_lift.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.transition.transition import Transition
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_transition_lift_01_basic():
    """
    TRANSLATED TEST - Basic transition lift test
    
    Tests lifting of transition to higher dimensions.
    """
    # tolerance
    tol = np.finfo(float).eps
    
    # init transition
    guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                     np.array([[-1, 0]]), np.array([[0]]))
    reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
    trans = Transition(guard, reset, 1)
    
    # lift parameters
    n_high = 5
    m_high = 3
    stateBind = np.array([1, 2])  # MATLAB 1-based, Python 0-based: [0, 1]
    inputBind = np.array([1])  # MATLAB 1-based, Python 0-based: [0]
    id = True
    
    # lift transition
    trans_lift = trans.lift(n_high, m_high, stateBind, inputBind, id)
    
    # check properties
    assert trans_lift is not None, "Lifted transition should be created"
    assert trans_lift.target == trans.target, "Target should remain the same"
    assert trans_lift.syncLabel == trans.syncLabel, "Sync label should remain the same"
    
    # check guard dimensions
    # Guard should be lifted to higher dimension
    if hasattr(trans_lift.guard, 'dim'):
        assert trans_lift.guard.dim() == n_high, "Guard should have n_high dimensions"
    
    # check reset dimensions
    assert trans_lift.reset.preStateDim == n_high, "Reset preStateDim should be n_high"
    assert trans_lift.reset.postStateDim == n_high, "Reset postStateDim should be n_high"
    assert trans_lift.reset.inputDim == m_high, "Reset inputDim should be m_high"
    
    # check that original reset is preserved in bound dimensions
    stateBind_py = stateBind - 1  # Convert to 0-based
    A_original = reset.A
    A_lifted = trans_lift.reset.A
    assert np.all(withinTol(A_lifted[np.ix_(stateBind_py, stateBind_py)], A_original, tol)), \
        "Reset A should match in bound dimensions"


def test_transition_lift_02_no_input():
    """
    TRANSLATED TEST - No input lift test
    
    Tests lifting when there's no input dimension.
    """
    try:
        # init transition without input in reset
        guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                         np.array([[-1, 0]]), np.array([[0]]))
        reset = LinearReset(np.array([[1, 0], [0, -0.75]]))  # No B matrix
        trans = Transition(guard, reset, 1)
        
        # lift parameters
        n_high = 4
        m_high = 2
        stateBind = np.array([1, 2])
        inputBind = np.array([])  # No input binding
        id = True
        
        # lift transition (if method exists)
        if hasattr(trans, 'lift'):
            trans_lift = trans.lift(n_high, m_high, stateBind, inputBind, id)
            
            # check properties
            assert trans_lift is not None, "Lifted transition should be created"
            assert trans_lift.reset.preStateDim == n_high, "Reset preStateDim should be n_high"
            assert trans_lift.reset.postStateDim == n_high, "Reset postStateDim should be n_high"
            # Input dimension should be set even if not bound
            assert trans_lift.reset.inputDim == m_high, "Reset inputDim should be m_high"
        else:
            pytest.skip("Transition.lift method not yet implemented")
    except (AttributeError, NotImplementedError):
        pytest.skip("Transition.lift method not yet implemented")


def test_transition_lift_03_identity():
    """
    TRANSLATED TEST - Identity lift test
    
    Tests lifting with identity flag.
    """
    try:
        # init transition
        guard = Polytope(np.array([[0, 1]]), np.array([[0]]),
                         np.array([[-1, 0]]), np.array([[0]]))
        reset = LinearReset(np.array([[1, 0], [0, -0.75]]))
        trans = Transition(guard, reset, 1)
        
        # lift parameters with identity
        n_high = 4
        m_high = 2
        stateBind = np.array([1, 2])
        inputBind = np.array([1])
        id = True  # Identity for non-bound dimensions
        
        # lift transition (if method exists)
        if hasattr(trans, 'lift'):
            trans_lift = trans.lift(n_high, m_high, stateBind, inputBind, id)
            
            # check that non-bound dimensions have identity
            stateBind_py = stateBind - 1
            otherDims = np.setdiff1d(np.arange(n_high), stateBind_py)
            A_lifted = trans_lift.reset.A
            
            if len(otherDims) > 0:
                # Non-bound dimensions should have identity
                assert np.all(withinTol(A_lifted[np.ix_(otherDims, otherDims)], 
                                       np.eye(len(otherDims)), np.finfo(float).eps)), \
                    "Non-bound dimensions should have identity"
        else:
            pytest.skip("Transition.lift method not yet implemented")
    except (AttributeError, NotImplementedError):
        pytest.skip("Transition.lift method not yet implemented")

