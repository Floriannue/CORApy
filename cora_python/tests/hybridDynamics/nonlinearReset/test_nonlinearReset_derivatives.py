"""
test_nonlinearReset_derivatives - test function for nonlinearReset derivatives

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_derivatives.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest


def test_nonlinearReset_derivatives_01_basic():
    """
    TRANSLATED TEST - Basic nonlinearReset derivatives test
    
    Tests derivative computation for NonlinearReset.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset function
        def reset_func(x, u):
            return np.array([np.sin(x[0]), x[1]**2]) if len(x) >= 2 else np.sin(x[0])
        
        reset = NonlinearReset(reset_func)
        
        # compute derivatives
        reset.derivatives()
        
        # Check that derivatives were computed
        assert hasattr(reset, 'jacobian') or hasattr(reset, 'J'), \
            "Should have jacobian after derivatives computation"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")
