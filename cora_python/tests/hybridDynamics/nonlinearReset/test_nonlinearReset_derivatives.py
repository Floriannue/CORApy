"""
test_nonlinearReset_derivatives - test function for nonlinearReset derivatives

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_derivatives.m

This test file is a DIRECT TRANSLATION from the MATLAB test file.
All test cases match the MATLAB implementation exactly, with additional test cases
added for comprehensive coverage (tensorOrder=3, default path, file generation).

Original MATLAB test covers:
- Basic derivatives with tensorOrder=2 (Jacobian + Hessian)
- Different function signatures (only states, states+inputs, different output dims)

Extended Python test adds:
- tensorOrder=1 (Jacobian only)
- tensorOrder=3 (all derivatives)
- Default path handling
- File generation verification

Authors:       Mark Wetzlinger
Written:       12-October-2024 (MATLAB)
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import os
import numpy as np
import pytest
import tempfile
import shutil
from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
from cora_python.g.macros.CORAROOT import CORAROOT


def test_nonlinearReset_derivatives_basic():
    """
    GENERATED TEST - Basic nonlinearReset derivatives test
    
    Tests derivative computation for NonlinearReset with tensorOrder=1 (Jacobian only).
    This test was generated to provide additional coverage beyond the MATLAB test.
    """
    # define reset function: f(x,u) = [x(1)*x(2); x(2)]
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        u_arr = np.asarray(u).flatten() if u is not None else np.array([0])
        return np.array([[x_arr[0]*x_arr[1]], [x_arr[1]]])
    
    reset = NonlinearReset(reset_func, 2, 1, 2)
    
    # create temporary directory for generated files
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = 'test_nonlinearReset_derivatives'
        
        # compute derivatives with tensorOrder=1 (only Jacobian)
        reset = reset.derivatives(temp_dir, fname, 1)
        
        # Check that derivatives were computed
        assert hasattr(reset, 'J'), "Should have J (Jacobian) after derivatives computation"
        assert reset.J is not None, "Jacobian should not be None"
        assert reset.tensorOrder == 1, "Tensor order should be 1"
        
        # H and T should not be computed for tensorOrder=1
        # (They may be None or not set)
        if hasattr(reset, 'H'):
            assert reset.H is None or reset.H == [], "Hessian should not be computed for tensorOrder=1"
        if hasattr(reset, 'T'):
            assert reset.T is None or reset.T == [], "Third-order tensor should not be computed for tensorOrder=1"


def test_nonlinearReset_derivatives_tensorOrder2_only_states():
    """
    TRANSLATED TEST - Test derivatives with tensorOrder=2, only states
    
    Translated from MATLAB: test_nonlinearReset_derivatives.m lines 44-51
    Original test: f = @(x,u) [x(1)*x(2); x(2)]
    """
    # define reset function: f(x,u) = [x(1)*x(2); x(2)]
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        u_arr = np.asarray(u).flatten() if u is not None else np.array([0])
        return np.array([[x_arr[0]*x_arr[1]], [x_arr[1]]])
    
    reset = NonlinearReset(reset_func)
    
    # create temporary directory for generated files
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = 'test_nonlinearReset_derivatives_generatedfile'
        
        # compute derivatives with tensorOrder=2 (Jacobian + Hessian)
        reset = reset.derivatives(temp_dir, fname, 2)
        
        # Check that derivatives were computed (MATLAB: assert(~isempty(nonlinReset.J)))
        assert hasattr(reset, 'J'), "Should have J (Jacobian) after derivatives computation"
        assert reset.J is not None, "Jacobian should not be None"
        # MATLAB: assert(~isempty(nonlinReset.H))
        assert hasattr(reset, 'H'), "Should have H (Hessian) after derivatives computation"
        assert reset.H is not None, "Hessian should not be None"


def test_nonlinearReset_derivatives_tensorOrder2_states_inputs():
    """
    TRANSLATED TEST - Test derivatives with tensorOrder=2, states and inputs
    
    Translated from MATLAB: test_nonlinearReset_derivatives.m lines 53-60
    Original test: f = @(x,u) [x(1) - u(1); x(1)*x(2)]
    """
    # define reset function: f(x,u) = [x(1) - u(1); x(1)*x(2)]
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        u_arr = np.asarray(u).flatten() if u is not None else np.array([0])
        return np.array([[x_arr[0] - u_arr[0]], [x_arr[0]*x_arr[1]]])
    
    reset = NonlinearReset(reset_func)
    
    # create temporary directory for generated files
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = 'test_nonlinearReset_derivatives_generatedfile'
        
        # compute derivatives with tensorOrder=2 (Jacobian + Hessian)
        reset = reset.derivatives(temp_dir, fname, 2)
        
        # Check that derivatives were computed (MATLAB: assert(~isempty(nonlinReset.J)))
        assert hasattr(reset, 'J'), "Should have J (Jacobian) after derivatives computation"
        assert reset.J is not None, "Jacobian should not be None"
        # MATLAB: assert(~isempty(nonlinReset.H))
        assert hasattr(reset, 'H'), "Should have H (Hessian) after derivatives computation"
        assert reset.H is not None, "Hessian should not be None"


def test_nonlinearReset_derivatives_tensorOrder2_different_output_dim():
    """
    TRANSLATED TEST - Test derivatives with tensorOrder=2, different output dimension
    
    Translated from MATLAB: test_nonlinearReset_derivatives.m lines 63-69
    Original test: f = @(x,u) x(1)*x(2) - u(1) (scalar output)
    """
    # define reset function: f(x,u) = x(1)*x(2) - u(1) (scalar output)
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        u_arr = np.asarray(u).flatten() if u is not None else np.array([0])
        return np.array([[x_arr[0]*x_arr[1] - u_arr[0]]])
    
    reset = NonlinearReset(reset_func)
    
    # create temporary directory for generated files
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = 'test_nonlinearReset_derivatives_generatedfile'
        
        # compute derivatives with tensorOrder=2 (Jacobian + Hessian)
        reset = reset.derivatives(temp_dir, fname, 2)
        
        # Check that derivatives were computed (MATLAB: assert(~isempty(nonlinReset.J)))
        assert hasattr(reset, 'J'), "Should have J (Jacobian) after derivatives computation"
        assert reset.J is not None, "Jacobian should not be None"
        # MATLAB: assert(~isempty(nonlinReset.H))
        assert hasattr(reset, 'H'), "Should have H (Hessian) after derivatives computation"
        assert reset.H is not None, "Hessian should not be None"


def test_nonlinearReset_derivatives_tensorOrder3():
    """
    GENERATED TEST - Test derivatives with tensorOrder=3 (Jacobian + Hessian + Third-order tensor)
    
    This test was generated to provide coverage for tensorOrder=3, which is not in the MATLAB test.
    """
    # define reset function: f(x,u) = [x(1)*x(2); x(2)]
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        u_arr = np.asarray(u).flatten() if u is not None else np.array([0])
        return np.array([[x_arr[0]*x_arr[1]], [x_arr[1]]])
    
    reset = NonlinearReset(reset_func, 2, 1, 2)
    
    # create temporary directory for generated files
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = 'test_nonlinearReset_derivatives_order3'
        
        # compute derivatives with tensorOrder=3 (all derivatives)
        reset = reset.derivatives(temp_dir, fname, 3)
        
        # Check that all derivatives were computed
        assert hasattr(reset, 'J'), "Should have J (Jacobian) after derivatives computation"
        assert reset.J is not None, "Jacobian should not be None"
        assert hasattr(reset, 'H'), "Should have H (Hessian) after derivatives computation"
        assert reset.H is not None, "Hessian should not be None"
        assert hasattr(reset, 'T'), "Should have T (Third-order tensor) after derivatives computation"
        assert reset.T is not None, "Third-order tensor should not be None"
        assert reset.tensorOrder == 3, "Tensor order should be 3"


def test_nonlinearReset_derivatives_default_path():
    """
    GENERATED TEST - Test derivatives with default path
    
    This test was generated to verify default path handling, which is not explicitly tested in MATLAB.
    """
    # define reset function: f(x,u) = [x(1); x(2)]
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        return np.array([[x_arr[0]], [x_arr[1]]])
    
    reset = NonlinearReset(reset_func, 2, 1, 2)
    
    # compute derivatives with default path (should use CORAROOT/models/auxiliary/nonlinearReset)
    reset = reset.derivatives()
    
    # Check that derivatives were computed
    assert hasattr(reset, 'J'), "Should have J (Jacobian) after derivatives computation"
    assert reset.J is not None, "Jacobian should not be None"
    assert reset.tensorOrder == 2, "Default tensor order should be 2"


def test_nonlinearReset_derivatives_file_generation():
    """
    GENERATED TEST - Test that derivative files are generated correctly
    
    This test was generated to verify file generation, which is not explicitly tested in MATLAB.
    """
    # define reset function: f(x,u) = [x(1)*x(2); x(2)]
    def reset_func(x, u):
        x_arr = np.asarray(x).flatten()
        u_arr = np.asarray(u).flatten() if u is not None else np.array([0])
        return np.array([[x_arr[0]*x_arr[1]], [x_arr[1]]])
    
    reset = NonlinearReset(reset_func, 2, 1, 2)
    
    # create temporary directory for generated files
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = 'test_derivatives_file_gen'
        
        # compute derivatives
        reset = reset.derivatives(temp_dir, fname, 1)
        
        # Check that files were generated
        jacobian_file = os.path.join(temp_dir, f'{fname}_jacobian.py')
        assert os.path.exists(jacobian_file), f"Jacobian file should be generated at {jacobian_file}"
        
        # Check that the file is importable (basic check)
        assert os.path.getsize(jacobian_file) > 0, "Jacobian file should not be empty"


if __name__ == '__main__':
    pytest.main([__file__])
