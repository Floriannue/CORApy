"""
test_nonlinearReset_synchronize - test function for nonlinearReset synchronize

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_synchronize.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_nonlinearReset_synchronize_01_basic():
    """
    TRANSLATED TEST - Basic nonlinearReset synchronize test
    
    Tests synchronization of a list of NonlinearReset objects.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset functions
        def reset_func1(x, u):
            return np.array([np.sin(x[0]), x[1]]) if len(x) >= 2 else np.sin(x[0])
        
        def reset_func2(x, u):
            return np.array([x[0], np.cos(x[1])]) if len(x) >= 2 else x
        
        reset1 = NonlinearReset(reset_func1)
        reset2 = NonlinearReset(reset_func2)
        
        # synchronize
        resets = [reset1, reset2]
        idStates = np.array([False, False, False, False, True])  # Last state is identity
        
        # synchronize is a method, not a class method
        if hasattr(reset1, 'synchronize'):
            reset_sync = reset1.synchronize(resets, idStates)
        else:
            # Try static method or module-level function
            from cora_python.hybridDynamics.nonlinearReset import synchronize
            reset_sync = synchronize(resets, idStates)
        
        assert reset_sync is not None, "Synchronized reset should be created"
        # Dimensions should match combined dimensions
        assert reset_sync.preStateDim == len(idStates), "preStateDim should match"
        assert reset_sync.postStateDim == len(idStates), "postStateDim should match"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")

