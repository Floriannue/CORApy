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
    # blkdiag equivalent: B1 is 3x2, B2 is 2x1, so result is 5x3
    B = np.block([[B1, np.zeros((3, 1))],
                  [np.zeros((2, 2)), B2]])
    c = np.zeros((5, 1))
    linReset = LinearReset(A, B, c)
    
    # binds and flows
    # Python uses 0-based indexing everywhere
    stateBinds = [np.array([0, 1, 2]), np.array([3, 4])]  # Python 0-based
    # inputBinds: each is (num_inputs, 2) array where:
    #   first column: component index (0=global, >=1=other component, 1-based for MATLAB compatibility)
    #   second column: output/input index (1-based for MATLAB compatibility)
    # Component 0: input 0 -> component 1 output 0, input 1 -> global input 0
    # Component 1: input 0 -> component 0 output 0
    # Note: inputBinds still uses MATLAB 1-based convention (1=component 0, 2=component 1)
    # The resolve method converts these to 0-based internally
    inputBinds = [np.array([[2, 1], [0, 1]]), np.array([[1, 1]])]  # MATLAB 1-based indices (converted internally)
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
    # Both sys1.k and sys2.k are scalars in this test
    # Convert to column vectors for matrix multiplication
    sys2_k_val = float(sys2.k) if hasattr(sys2.k, '__float__') else sys2.k
    sys2_k = np.array([[sys2_k_val]])
    sys1_k_val = float(sys1.k) if hasattr(sys1.k, '__float__') else sys1.k
    sys1_k = np.array([[sys1_k_val]])
    c_resolved = np.vstack([B1[:, 0:1] @ sys2_k, B2 @ sys1_k])
    assert compareMatrices(linReset_.c, c_resolved, tol, "equal", True), \
        "c should match resolved vector"

