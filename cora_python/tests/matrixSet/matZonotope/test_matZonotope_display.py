"""
test_matZonotope_display - unit test function for display (only check for
   runtime errors)

TRANSLATED FROM: cora_matlab/unitTests/matrixSet/matZonotope/test_matZonotope_display.m

Syntax:
    pytest cora_python/tests/matrixSet/matZonotope/test_matZonotope_display.py

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
from cora_python.matrixSet.matZonotope import matZonotope
import io
import sys


def test_matZonotope_display_empty():
    """
    TRANSLATED TEST - Empty matrix zonotope
    """
    # MATLAB: matZ = matZonotope()
    matZ = matZonotope()
    
    display_str = matZ.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matZonotope_display_scalar():
    """
    TRANSLATED TEST - Scalar
    """
    # MATLAB: C = 0; G = []; G(:,:,1) = 1; G(:,:,2) = -2;
    C = np.array([[0]])
    G = np.zeros((1, 1, 2))
    G[:, :, 0] = np.array([[1]])
    G[:, :, 1] = np.array([[-2]])
    matZ = matZonotope(C, G)
    
    display_str = matZ.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matZonotope_display_nx1_vector():
    """
    TRANSLATED TEST - nx1 vector
    """
    # MATLAB: C = [0; 1; 1];
    # MATLAB: G = []; G(:,:,1) = [1; -1; -2]; G(:,:,2) = [-2; 0; 1];
    C = np.array([[0], [1], [1]])
    G = np.zeros((3, 1, 2))
    G[:, :, 0] = np.array([[1], [-1], [-2]])
    G[:, :, 1] = np.array([[-2], [0], [1]])
    matZ = matZonotope(C, G)
    
    display_str = matZ.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matZonotope_display_matrix():
    """
    TRANSLATED TEST - Matrix
    """
    # MATLAB: C = [0 2; 1 -1; 1 -2];
    # MATLAB: G = []; G(:,:,1) = [1 1; -1 0; -2 1]; G(:,:,2) = [-2 0; 0 1; 1 -1];
    C = np.array([[0, 2], [1, -1], [1, -2]])
    G = np.zeros((3, 2, 2))
    G[:, :, 0] = np.array([[1, 1], [-1, 0], [-2, 1]])
    G[:, :, 1] = np.array([[-2, 0], [0, 1], [1, -1]])
    matZ = matZonotope(C, G)
    
    display_str = matZ.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matZonotope_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    C = np.array([[0, 2], [1, -1]])
    G = np.zeros((2, 2, 1))
    G[:, :, 0] = np.array([[1, 1], [-1, 0]])
    matZ = matZonotope(C, G)
    
    display_str = matZ.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        matZ.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(matZ) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

