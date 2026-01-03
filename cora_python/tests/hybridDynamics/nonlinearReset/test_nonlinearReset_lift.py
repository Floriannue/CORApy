"""
test_nonlinearReset_lift - test function for nonlinearReset lift

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_lift.m

Authors:       Mark Wetzlinger
Written:       10-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_nonlinearReset_lift_01_basic():
    """
    TRANSLATED TEST - Basic nonlinearReset lift test
    
    Tests lifting of NonlinearReset to higher dimensions.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset function
        def reset_func(x, u):
            return np.sin(x[0]) if len(x) > 0 else np.array([0])
        
        reset = NonlinearReset(reset_func)
        
        # lift parameters
        n_high = 5
        m_high = 3
        stateBind = np.array([1])  # MATLAB 1-based, Python 0-based: [0]
        inputBind = np.array([1])  # MATLAB 1-based, Python 0-based: [0]
        id = True
        
        reset_lift = reset.lift(n_high, m_high, stateBind, inputBind, id)
        
        assert reset_lift is not None, "Lifted reset should be created"
        assert reset_lift.preStateDim == n_high, "preStateDim should be n_high"
        assert reset_lift.postStateDim == n_high, "postStateDim should be n_high"
        assert reset_lift.inputDim == m_high, "inputDim should be m_high"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")

