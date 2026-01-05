"""
test_intervalMatrix_display - unit test function for display (only check
    for runtime errors)

TRANSLATED FROM: cora_matlab/unitTests/matrixSet/intervalMatrix/test_intervalMatrix_display.m

Syntax:
    pytest cora_python/tests/matrixSet/intervalMatrix/test_intervalMatrix_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.matrixSet.intervalMatrix.intervalMatrix import IntervalMatrix
import io
import sys


def test_intervalMatrix_display_scalar():
    """
    TRANSLATED TEST - Scalar intervalMatrix display test
    """
    # scalar
    intMat = IntervalMatrix(1, 0)
    
    display_str = intMat.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        intMat.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout


def test_intervalMatrix_display_vector():
    """
    TRANSLATED TEST - Vector intervalMatrix display test
    """
    # vector
    c = np.array([[1], [0], [1]])
    d = np.array([[1], [2], [2]])
    intMat = IntervalMatrix(c, d)
    
    display_str = intMat.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_intervalMatrix_display_matrix():
    """
    TRANSLATED TEST - Matrix intervalMatrix display test
    """
    # matrix
    c = np.array([[2, 3, 4], [5, 6, 0]])
    d = np.array([[1, 0, 1], [0, 0, 1]])
    intMat = IntervalMatrix(c, d)
    
    display_str = intMat.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_intervalMatrix_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    intMat = IntervalMatrix(1, 0)
    
    display_str = intMat.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        intMat.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(intMat) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

