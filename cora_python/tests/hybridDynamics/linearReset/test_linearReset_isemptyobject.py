"""
test_linearReset_isemptyobject - test function for isemptyobject

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/linearReset/test_linearReset_isemptyobject.m

Authors:       Mark Wetzlinger
Written:       15-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset


def test_linearReset_isemptyobject_01_empty():
    """
    TRANSLATED TEST - Empty linearReset test
    """
    # empty object
    assert LinearReset().isemptyobject(), "Empty linearReset should be empty"


def test_linearReset_isemptyobject_02_nonempty():
    """
    TRANSLATED TEST - Non-empty linearReset test
    """
    # linearReset
    A = np.array([[0, 0], [0, 0.2]])
    reset = LinearReset(A)
    
    assert not reset.isemptyobject(), "Non-empty linearReset should not be empty"
    
    # with B and c
    B = np.array([[0], [1]])
    c = np.array([[1], [0]])
    reset2 = LinearReset(A, B, c)
    assert not reset2.isemptyobject(), "Non-empty linearReset with B and c should not be empty"
    
    # array of linearResets
    reset_array = [LinearReset(), LinearReset(A)]
    empty_results = [r.isemptyobject() for r in reset_array]
    assert empty_results == [True, False], \
        "Array of linearResets should have correct empty results"

