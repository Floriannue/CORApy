"""
test_linearReset_resolve - test function for the resolution of local
   inputs to states of other components (via outputs) and global inputs

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_resolve.m

Authors:       Mark Wetzlinger
Written:       14-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


def test_linearReset_resolve_01_basic():
    """
    TRANSLATED TEST - Basic resolve test
    
    Tests resolution of local inputs to states of other components.
    """
    # tolerance
    tol = np.finfo(float).eps
    
    # init synchronized linear reset function
    A = np.zeros((5, 5))
    B1 = np.array([[1, -1], [2, -2], [3, -3]])
    B2 = np.array([[4], [5]])
    # blkdiag equivalent
    B = np.block([[B1, np.zeros((3, 1))],
                  [np.zeros((2, 3)), B2]])
    c = np.zeros((5, 1))
    linReset = LinearReset(A, B, c)
    
    # binds and flows
    stateBinds = [[1, 2, 3], [4, 5]]  # MATLAB 1-based, Python 0-based: [[0,1,2], [3,4]]
    inputBinds = [[2, 1, 0, 1], [1, 1]]  # Note: This structure might need adjustment
    sys1 = LinearSys('linearSys', np.eye(3), np.zeros((3, 2)), np.zeros((3, 1)),
                     np.array([1, 2, 3]), np.array([0, 100]), 5)
    sys2 = LinearSys('linearSys', np.eye(2), np.zeros((2, 1)), np.zeros((2, 1)),
                     np.array([-1, -2]), 0, -10)
    flowList = [sys1, sys2]
    
    # resolve input binds
    linReset_ = linReset.resolve(flowList, stateBinds, inputBinds)
    
    # A: affected by resolution of u11 = y21 and u21 = y11
    A_resolved = np.block([[np.zeros((3, 3)), B1[:, 0:1] @ sys2.C],
                           [B2 @ sys1.C, np.zeros((2, 2))]])
    assert compareMatrices(linReset_.A, A_resolved, tol, "equal", True), \
        "A should match resolved matrix"
    
    # B: only part with global inputs remains
    B_resolved = np.vstack([B1[:, 1:2], B2 @ sys1.D[:, 1:2]])
    assert compareMatrices(linReset_.B, B_resolved, tol, "equal", True), \
        "B should match resolved matrix"
    
    # c: affected by resolution of u11 = y21 and u21 = y11
    c_resolved = np.vstack([B1[:, 0:1] @ sys2.k, B2 @ sys1.k])
    assert compareMatrices(linReset_.c, c_resolved, tol, "equal", True), \
        "c should match resolved vector"

