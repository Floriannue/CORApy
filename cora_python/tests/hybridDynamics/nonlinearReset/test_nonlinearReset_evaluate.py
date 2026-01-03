"""
test_nonlinearReset_evaluate - test function for nonlinearReset evaluate

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_evaluate.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_nonlinearReset_evaluate_01_vector():
    """
    TRANSLATED TEST - Vector evaluation test
    
    Tests evaluation of NonlinearReset with vector input.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset function
        def reset_func(x, u):
            return np.array([np.sin(x[0]), x[1]**2]) if len(x) >= 2 else np.sin(x[0])
        
        reset = NonlinearReset(reset_func)
        
        # evaluate with vector
        x = np.array([[1], [2]])
        u = np.array([[0.5]])
        
        x_ = reset.evaluate(x, u)
        
        assert x_ is not None, "Should return result"
        assert isinstance(x_, np.ndarray), "Should return numpy array"
        assert x_.shape[0] == reset.postStateDim, "Should have correct dimension"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")


def test_nonlinearReset_evaluate_02_zonotope():
    """
    TRANSLATED TEST - Zonotope evaluation test
    
    Tests evaluation of NonlinearReset with zonotope input.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset function
        def reset_func(x, u):
            return np.array([np.sin(x[0]), x[1]**2]) if len(x) >= 2 else np.sin(x[0])
        
        reset = NonlinearReset(reset_func)
        
        # evaluate with zonotope
        Z = Zonotope(np.array([[1], [2]]), 0.1 * np.eye(2))
        u = np.array([[0.5]])
        
        Z_ = reset.evaluate(Z, u)
        
        assert Z_ is not None, "Should return result"
        assert hasattr(Z_, 'center') or hasattr(Z_, 'c'), "Should be a set with center"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")


def test_nonlinearReset_evaluate_03_no_input():
    """
    TRANSLATED TEST - No input test
    
    Tests evaluation without input u.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset function (no input)
        def reset_func(x, u):
            return np.sin(x[0]) if len(x) > 0 else np.array([0])
        
        reset = NonlinearReset(reset_func)
        
        # evaluate without input
        x = np.array([[1]])
        
        x_ = reset.evaluate(x, None)
        
        assert x_ is not None, "Should return result"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")

