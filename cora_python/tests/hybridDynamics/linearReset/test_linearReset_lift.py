"""
test_linearReset_lift - test function for the projection of a linear
   reset function

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_lift.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_linearReset_lift_01_basic():
    """
    TRANSLATED TEST - Basic lift test
    
    Tests lifting of linear reset functions to higher dimensions.
    """
    # tolerance
    tol = np.finfo(float).eps
    
    # init linear reset functions
    A = np.array([[1, 2], [0, -1]])
    B = np.array([[2, 0, 1], [-1, 0, 0]])
    c = np.array([[1], [-5]])
    linReset_A = LinearReset(A)
    linReset_AB = LinearReset(A, B)
    linReset_ABc = LinearReset(A, B, c)
    
    # lift
    n_high = 6
    stateBind = np.array([1, 2])  # Python 0-based indices
    m_high = 5
    m_high_noB = 3
    inputBind = np.array([1, 2, 3])  # Python 0-based indices
    inputBind_noB = 1  # Python 0-based index
    id = True
    
    linReset_A_lift = linReset_A.lift(n_high, m_high_noB, stateBind, inputBind_noB, id)
    linReset_AB_lift = linReset_AB.lift(n_high, m_high, stateBind, inputBind, id)
    linReset_ABc_lift = linReset_ABc.lift(n_high, m_high, stateBind, inputBind, id)
    
    # check projected dimensions
    # stateBind is already 0-based
    stateBind_py = stateBind
    assert np.all(withinTol(linReset_A_lift.A[np.ix_(stateBind_py, stateBind_py)], A, tol)), \
        "A should match in projected dimensions"
    assert np.all(withinTol(linReset_AB_lift.A[np.ix_(stateBind_py, stateBind_py)], A, tol)), \
        "A should match in projected dimensions"
    assert np.all(withinTol(linReset_ABc_lift.A[np.ix_(stateBind_py, stateBind_py)], A, tol)), \
        "A should match in projected dimensions"
    
    # check new dimensions
    otherDims = np.setdiff1d(np.arange(n_high), stateBind_py)
    n_plus = n_high - len(stateBind)
    assert np.all(withinTol(linReset_A_lift.A[np.ix_(otherDims, otherDims)], np.eye(n_plus), tol)), \
        "Other dimensions should be identity"
    assert np.all(withinTol(linReset_AB_lift.A[np.ix_(otherDims, otherDims)], np.eye(n_plus), tol)), \
        "Other dimensions should be identity"
    assert np.all(withinTol(linReset_ABc_lift.A[np.ix_(otherDims, otherDims)], np.eye(n_plus), tol)), \
        "Other dimensions should be identity"
    
    # check input matrix
    assert np.all(withinTol(linReset_A_lift.B, np.zeros((n_high, 1)), tol)), \
        "B should be zero for reset without B"
    inputBind_py = inputBind  # Already 0-based
    assert np.all(withinTol(linReset_AB_lift.B[np.ix_(stateBind_py, inputBind_py)], B, tol)), \
        "B should match in projected dimensions"
    assert np.all(withinTol(linReset_ABc_lift.B[np.ix_(stateBind_py, inputBind_py)], B, tol)), \
        "B should match in projected dimensions"
    
    # check vector
    assert np.all(withinTol(linReset_A_lift.c, np.zeros((n_high, 1)), tol)), \
        "c should be zero for reset without c"
    assert np.all(withinTol(linReset_AB_lift.c, np.zeros((n_high, 1)), tol)), \
        "c should be zero for reset without c"
    assert np.all(withinTol(linReset_ABc_lift.c[stateBind_py], c, tol)), \
        "c should match in projected dimensions"

