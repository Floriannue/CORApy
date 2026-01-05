"""
test_nonlinearSysDT_display - unit test for display function

TRANSLATED FROM: cora_matlab/unitTests/contDynamics/nonlinearSysDT/test_nonlinearSysDT_display.m

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSysDT/test_nonlinearSysDT_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-November-2022 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contDynamics.nonlinearSysDT.nonlinearSysDT import NonlinearSysDT
import io
import sys


def test_nonlinearSysDT_display_one_dim_no_inputs():
    """
    TRANSLATED TEST - One-dimensional, no inputs, no outputs
    """
    dt = 1
    f = lambda x, u: x[0]**2
    sys = NonlinearSysDT(f, dt)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_one_dim_with_inputs():
    """
    TRANSLATED TEST - One-dimensional, with inputs, no outputs
    """
    dt = 1
    f = lambda x, u: x[0]**2 * u[0] - np.exp(x[0])
    sys = NonlinearSysDT(f, dt, states=1, inputs=1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_one_dim_linear_output():
    """
    TRANSLATED TEST - One-dimensional, with inputs, linear output
    """
    dt = 1
    f = lambda x, u: x[0]**2 * u[0] - np.exp(x[0])
    g = lambda x, u: x[0] - u[0]
    sys = NonlinearSysDT(f, dt, g, states=1, inputs=1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_one_dim_nonlinear_output():
    """
    TRANSLATED TEST - One-dimensional, with inputs, nonlinear output
    """
    dt = 1
    f = lambda x, u: x[0]**2 * u[0] - np.exp(x[0])
    g = lambda x, u: np.sin(x[0]) - u[0]
    sys = NonlinearSysDT(f, dt, g, states=1, inputs=1)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_multi_dim_no_inputs():
    """
    TRANSLATED TEST - Multi-dimensional, no inputs
    """
    dt = 1
    f = lambda x, u: np.array([[x[0]**2 + x[1]], [x[1] - np.exp(x[0])]])
    sys = NonlinearSysDT(f, dt, states=2, inputs=0)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_multi_dim_with_inputs():
    """
    TRANSLATED TEST - Multi-dimensional, with inputs
    """
    dt = 1
    f = lambda x, u: np.array([[x[0]**2 + x[1] - u[0]], [x[1] - np.exp(x[0]) + u[1]]])
    sys = NonlinearSysDT(f, dt, states=2, inputs=2)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_multi_dim_linear_output():
    """
    TRANSLATED TEST - Multi-dimensional, with inputs, linear output
    """
    dt = 1
    f = lambda x, u: np.array([[x[0]**2 + x[1] - u[0]], [x[1] - np.exp(x[0]) + u[1]]])
    g = lambda x, u: x[1] - x[0]
    sys = NonlinearSysDT(f, dt, g, states=2, inputs=2)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_multi_dim_nonlinear_output():
    """
    TRANSLATED TEST - Multi-dimensional, with inputs, nonlinear output
    """
    dt = 1
    f = lambda x, u: np.array([[x[0]**2 + x[1] - u[0]], [x[1] - np.exp(x[0]) + u[1]]])
    g = lambda x, u: np.array([[x[1] * x[0]], [u[0] * np.sqrt(x[0])]])
    sys = NonlinearSysDT(f, dt, g, states=2, inputs=2)
    
    display_str = sys.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_nonlinearSysDT_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    dt = 1
    f = lambda x, u: x[0]**2
    nonlinear_sys = NonlinearSysDT(f, dt)
    
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

