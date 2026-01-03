"""
test_linearReset_synchronize - test function for the synchronization of
   a list of linear reset functions

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_synchronize.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
from scipy.linalg import block_diag


def test_linearReset_synchronize_01_basic():
    """
    TRANSLATED TEST - Basic synchronize test
    
    Tests synchronization of a list of linear reset functions.
    """
    # tolerance
    tol = np.finfo(float).eps
    
    # init linear reset functions
    A1 = block_diag(np.array([[1, 2], [0, -1]]), np.zeros((3, 3)))
    B1 = block_diag(np.array([[2, 0, 1], [-1, 0, 0]]), np.zeros((3, 1)))
    c1 = np.vstack([np.array([[1], [-5]]), np.zeros((3, 1))])
    linReset1 = LinearReset(A1, B1, c1)
    
    A2 = block_diag(np.zeros((2, 2)), np.array([[-1, 0], [5, 3]]), np.array([[0]]))
    B2 = block_diag(np.zeros((2, 3)), np.array([[-1], [0], [0]]))
    c2 = np.vstack([np.zeros((2, 1)), np.array([[1], [-2], [0]])])
    linReset2 = LinearReset(A2, B2, c2)
    
    # synchronize
    linResets = [linReset1, linReset2]
    idStates = np.array([False, False, False, False, True])
    linReset_sync = LinearReset.synchronize(linResets, idStates)
    
    A_true = A1 + A2 + np.diag([0, 0, 0, 0, 1])
    B_true = B1 + B2
    c_true = c1 + c2
    linReset_sync_true = LinearReset(A_true, B_true, c_true)
    
    assert linReset_sync.isequal(linReset_sync_true, tol), \
        "Synchronized reset should match expected result"

