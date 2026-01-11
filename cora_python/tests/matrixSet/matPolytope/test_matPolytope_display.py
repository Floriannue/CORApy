"""
test_matPolytope_display - unit test function for display (only check for
   runtime errors)

TRANSLATED FROM: cora_matlab/unitTests/matrixSet/matPolytope/test_matPolytope_display.m

Syntax:
    pytest cora_python/tests/matrixSet/matPolytope/test_matPolytope_display.py

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
from cora_python.matrixSet.matPolytope import MatPolytope
import io
import sys


def test_matPolytope_display_empty():
    """
    TRANSLATED TEST - Empty matrix polytope
    """
    # MATLAB: matP = matPolytope()
    matP = MatPolytope()
    
    display_str = matP.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matPolytope_display_scalar():
    """
    TRANSLATED TEST - Scalar
    """
    # MATLAB: V = []; V(:,:,1) = 1; V(:,:,2) = -2;
    V = np.zeros((1, 1, 2))
    V[:, :, 0] = np.array([[1]])
    V[:, :, 1] = np.array([[-2]])
    matP = MatPolytope(V)
    
    display_str = matP.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matPolytope_display_nx1_vector():
    """
    TRANSLATED TEST - nx1 vector
    """
    # MATLAB: V = []; V(:,:,1) = [0; 1; 1]; V(:,:,2) = [1; -1; -2]; V(:,:,3) = [-2; 0; 1];
    V = np.zeros((3, 1, 3))
    V[:, :, 0] = np.array([[0], [1], [1]])
    V[:, :, 1] = np.array([[1], [-1], [-2]])
    V[:, :, 2] = np.array([[-2], [0], [1]])
    matP = MatPolytope(V)
    
    display_str = matP.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matPolytope_display_matrix():
    """
    TRANSLATED TEST - Matrix
    """
    # MATLAB: V = [];
    # MATLAB: V(:,:,1) = [0 2; 1 -1; 1 -2];
    # MATLAB: V(:,:,2) = [1 1; -1 0; -2 1];
    # MATLAB: V(:,:,3) = [-2 0; 0 1; 1 -1];
    V = np.zeros((3, 2, 3))
    V[:, :, 0] = np.array([[0, 2], [1, -1], [1, -2]])
    V[:, :, 1] = np.array([[1, 1], [-1, 0], [-2, 1]])
    V[:, :, 2] = np.array([[-2, 0], [0, 1], [1, -1]])
    matP = MatPolytope(V)
    
    display_str = matP.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_matPolytope_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    V = np.zeros((2, 2, 2))
    V[:, :, 0] = np.array([[0, 2], [1, -1]])
    V[:, :, 1] = np.array([[1, 1], [-1, 0]])
    matP = MatPolytope(V)
    
    display_str = matP.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        matP.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(matP) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

