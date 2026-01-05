"""
test_nonlinearARX_display - unit test for display function

TRANSLATED FROM: cora_matlab/unitTests/contDynamics/nonlinearARX/test_nonlinearARX_display.m

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearARX/test_nonlinearARX_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Laura Luetzow (MATLAB)
               Python translation by AI Assistant
Written:       15-May-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contDynamics.nonlinearARX.nonlinearARX import NonlinearARX
import io
import sys


def test_nonlinearARX_display_one_dim_no_inputs():
    """
    TRANSLATED TEST - One-dimensional, no inputs, past value
    """
    dt = 0.1
    f = lambda y, u: 0.5 * np.cos(y[0, 0])
    sys = NonlinearARX(f, dt, 1, 0, 1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearARX_display_one_dim_with_input():
    """
    TRANSLATED TEST - One-dimensional, with input, past value
    """
    dt = 0.1
    f = lambda y, u: 0.5 * np.cos(y[0, 0]) + u[0, 0] + u[1, 0]**2
    sys = NonlinearARX(f, dt, 1, 1, 1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearARX_display_one_dim_with_input_past_values():
    """
    TRANSLATED TEST - One-dimensional, with input, past values
    """
    dt = 0.1
    f = lambda y, u: 0.5 * np.cos(y[0, 0]) + u[0, 0] + y[1, 0]**2 - np.sin(u[2, 0])
    sys = NonlinearARX(f, dt, 1, 1, 2)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearARX_display_multi_dim_no_inputs():
    """
    TRANSLATED TEST - Multi-dimensional, no inputs, past value
    """
    dt = 0.1
    f = lambda y, u: np.array([[0.5 * np.cos(y[0, 0])], [y[1, 0]**2]])
    sys = NonlinearARX(f, dt, 2, 0, 1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearARX_display_multi_dim_with_input():
    """
    TRANSLATED TEST - Multi-dimensional, with input, past value
    """
    dt = 0.1
    f = lambda y, u: np.array([[0.5 * np.cos(y[1, 0]) + u[0, 0] - np.sin(u[2, 0])], 
                                [y[1, 0] - u[1, 0]**2]])
    sys = NonlinearARX(f, dt, 2, 2, 1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearARX_display_multi_dim_with_input_past_values():
    """
    TRANSLATED TEST - Multi-dimensional, with input, past values
    """
    dt = 0.1
    f = lambda y, u: np.array([[0.5 * np.cos(y[0, 0]) + u[0, 0] - np.sin(y[2, 0])], 
                                [y[1, 0]**2 - np.sin(u[2, 0] + u[4, 0])]])
    sys = NonlinearARX(f, dt, 2, 2, 2)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearARX_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    dt = 0.1
    f = lambda y, u: 0.5 * np.cos(y[0, 0])
    nonlinear_sys = NonlinearARX(f, dt, 1, 0, 1)
    
    display_str = nonlinear_sys.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        nonlinear_sys.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(nonlinear_sys) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

