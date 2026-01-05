"""
test_spectraShadow_display - unit test function of display

TRANSLATED FROM: cora_matlab/unitTests/contSet/spectraShadow/test_spectraShadow_display.m

Syntax:
    pytest cora_python/tests/contSet/spectraShadow/test_spectraShadow_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       14-August-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow
import io
import sys


def test_spectraShadow_display_empty():
    """
    TRANSLATED TEST - Empty spectrahedron display test
    """
    # empty spectrahedron
    SpS_empty = SpectraShadow.empty(1)
    
    # Should not raise an error
    display_str = SpS_empty.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        SpS_empty.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout


def test_spectraShadow_display_1d_cases():
    """
    TRANSLATED TEST - 1D spectraShadow cases
    """
    # 1D, bounded, non-degenerate
    SpS = SpectraShadow(np.array([[1, 0, 1, 0], [0, 1, 0, -1]]))
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # 1D, empty
    SpS = SpectraShadow(np.array([[-1, 0]]))
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    
    # 1D, unbounded
    SpS = SpectraShadow(np.array([[1, 0]]))
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    
    # 1D, single point
    SpS = SpectraShadow(np.array([[-1, 0, 1, 0], [0, 1, 0, -1]]))
    display_str = SpS.display_()
    assert isinstance(display_str, str)


def test_spectraShadow_display_2d_cases():
    """
    TRANSLATED TEST - 2D spectraShadow cases
    """
    # 2D, bounded, non-degenerate
    # MATLAB: A0 = eye(4); Ai{1} = blkdiag([1 0;0 -1],zeros(2)); Ai{2} = blkdiag(zeros(2),[1 0;0 -1]);
    # MATLAB: SpS = spectraShadow([A0 Ai{1} Ai{2}])
    # In MATLAB, [A0 Ai{1} Ai{2}] horizontally concatenates matrices
    A0 = np.eye(4)
    Ai1 = np.block([[np.array([[1, 0], [0, -1]]), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.zeros((2, 2))]])
    Ai2 = np.block([[np.zeros((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.array([[1, 0], [0, -1]])]])
    # Horizontally concatenate: [A0, Ai1, Ai2]
    A = np.hstack([A0, Ai1, Ai2])
    SpS = SpectraShadow(A)
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # 2D, bounded, degenerate
    A0 = np.block([[np.array([[-1, 0], [0, 1]]), np.zeros((2, 2))],
                   [np.zeros((2, 2)), np.array([[-1, 0], [0, 1]])]])
    Ai1 = np.block([[np.array([[1, 0], [0, -1]]), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.zeros((2, 2))]])
    Ai2 = np.block([[np.zeros((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.array([[1, 0], [0, -1]])]])
    A = np.hstack([A0, Ai1, Ai2])
    SpS = SpectraShadow(A)
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    
    # 2D, unbounded, non-degenerate
    SpS = SpectraShadow(np.array([[1, 0, 0]]))
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    
    # 2D, unbounded, degenerate
    A0 = np.array([[-1, 0], [0, 1]])
    Ai1 = np.zeros((2, 2))
    Ai2 = np.array([[1, 0], [0, -1]])
    A = np.hstack([A0, Ai1, Ai2])
    SpS = SpectraShadow(A)
    display_str = SpS.display_()
    assert isinstance(display_str, str)
    
    # 2D, empty
    SpS = SpectraShadow(np.array([[-1, 0, 0]]))
    display_str = SpS.display_()
    assert isinstance(display_str, str)


def test_spectraShadow_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    SpS = SpectraShadow(np.array([[1, 0, 1, 0], [0, 1, 0, -1]]))
    
    display_str = SpS.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        SpS.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(SpS) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

