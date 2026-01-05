"""
test_linearARX_display - unit test for display function

TRANSLATED FROM: cora_matlab/unitTests/contDynamics/linearARX/test_linearARX_display.m

Syntax:
    pytest cora_python/tests/contDynamics/linearARX/test_linearARX_display.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Laura Luetzow (MATLAB)
               Python translation by AI Assistant
Written:       05-July-2024 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contDynamics.linearARX.linearARX import LinearARX
import io
import sys


def test_linearARX_display():
    """
    TRANSLATED TEST - LinearARX display test
    """
    # initialize system and display
    A_bar = [np.array([[-0.4, 0.6], [0.6, -0.4]]), np.array([[0.1, 0], [0.2, -0.5]])]
    B_bar = [np.array([[0], [0]]), np.array([[0.3], [-0.7]]), np.array([[0.1], [0]])]
    dt = 0.1
    linear_sys = LinearARX('exampleSys', A_bar, B_bar, dt)
    
    # Should not raise an error
    display_str = linear_sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        linear_sys.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(linear_sys) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

