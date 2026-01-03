"""
test_linearReset_eye - test function for eye (identity reset)

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_eye.m

Authors:       Mark Wetzlinger
Written:       19-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset


def test_linearReset_eye_01_2D():
    """
    TRANSLATED TEST - 2D identity reset test
    """
    # 2D identity reset
    reset = LinearReset.eye(2)
    
    # check properties
    assert reset.A.shape == (2, 2), "A should be 2x2"
    np.testing.assert_array_equal(reset.A, np.eye(2), "A should be identity matrix")
    assert reset.B is None or reset.B.shape[0] == 2, "B should be None or have 2 rows"
    assert reset.c is None or reset.c.shape[0] == 2, "c should be None or have 2 rows"


def test_linearReset_eye_02_3D():
    """
    TRANSLATED TEST - 3D identity reset test
    """
    # 3D identity reset
    reset = LinearReset.eye(3)
    
    # check properties
    assert reset.A.shape == (3, 3), "A should be 3x3"
    np.testing.assert_array_equal(reset.A, np.eye(3), "A should be identity matrix")
    assert reset.preStateDim == 3, "preStateDim should be 3"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == 3, "postStateDim should be 3"


def test_linearReset_eye_03_states_and_inputs():
    """
    TRANSLATED TEST - States and inputs identity reset test
    """
    # states and inputs
    n = 3
    m = 2
    reset = LinearReset.eye(n, m)
    
    # check properties
    assert reset.preStateDim == n, "preStateDim should match n"
    assert reset.inputDim == m, "inputDim should match m"
    assert reset.postStateDim == n, "postStateDim should match n"


def test_linearReset_eye_04_empty():
    """
    TRANSLATED TEST - Empty states identity reset test
    """
    # empty states
    n = 0
    reset = LinearReset.eye(n)
    
    # check properties
    assert reset.preStateDim == n, "preStateDim should be 0"
    assert reset.inputDim == 1, "inputDim should be 1"
    assert reset.postStateDim == n, "postStateDim should be 0"
