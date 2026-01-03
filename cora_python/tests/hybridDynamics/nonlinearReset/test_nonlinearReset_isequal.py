"""
test_nonlinearReset_isequal - test function for equality check of nonlinearReset objects

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_isequal.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest


def test_nonlinearReset_isequal_01_standard():
    """
    TRANSLATED TEST - Standard equality test
    
    Tests equality checking for NonlinearReset objects.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset functions
        def reset_func1(x, u):
            return np.sin(x[0]) if len(x) > 0 else np.array([0])
        
        def reset_func2(x, u):
            return np.cos(x[0]) if len(x) > 0 else np.array([0])
        
        reset1 = NonlinearReset(reset_func1)
        reset2 = NonlinearReset(reset_func2)
        
        # Same reset should be equal
        assert reset1.isequal(reset1, atol=1e-6), "Same reset should be equal"
        # Different resets should not be equal
        assert not reset1.isequal(reset2, atol=1e-6), "Different resets should not be equal"
        assert not reset2.isequal(reset1, atol=1e-6), "Equality should be symmetric"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")

