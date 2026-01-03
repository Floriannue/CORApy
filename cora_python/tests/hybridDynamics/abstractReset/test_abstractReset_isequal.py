"""
test_abstractReset_isequal - test function for equality check of abstractReset objects

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/abstractReset/test_abstractReset_isequal.m

Authors:       Mark Wetzlinger
Written:       09-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.hybridDynamics.abstractReset.abstractReset import AbstractReset


def test_abstractReset_isequal_01_standard():
    """
    TRANSLATED TEST - Standard equality test
    
    Tests equality checking for AbstractReset objects.
    """
    # standard case
    reset1 = AbstractReset(1, 2, 1)
    reset2 = AbstractReset(1, 2, 2)  # Different postStateDim
    
    # Same reset should be equal
    assert reset1.isequal(reset1), "Same reset should be equal"
    # Different resets should not be equal
    assert not reset1.isequal(reset2), "Different resets should not be equal"
    assert not reset2.isequal(reset1), "Equality should be symmetric"


def test_abstractReset_isequal_02_operators():
    """
    TRANSLATED TEST - Operator tests
    
    Tests equality operators (==, !=) if implemented.
    """
    reset1 = AbstractReset(1, 2, 1)
    reset2 = AbstractReset(1, 2, 2)
    
    # Test == operator if implemented
    if hasattr(reset1, '__eq__'):
        assert reset1 == reset1, "Same reset should be equal with == operator"
        assert not (reset1 == reset2), "Different resets should not be equal with == operator"
        assert not (reset2 == reset1), "Equality should be symmetric"
    
    # Test != operator if implemented
    if hasattr(reset1, '__ne__'):
        assert not (reset1 != reset1), "Same reset should not be unequal with != operator"
        assert reset1 != reset2, "Different resets should be unequal with != operator"
        assert reset2 != reset1, "Inequality should be symmetric"

