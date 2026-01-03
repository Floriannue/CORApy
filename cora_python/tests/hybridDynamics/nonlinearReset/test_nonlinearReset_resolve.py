"""
test_nonlinearReset_resolve - test function for nonlinearReset resolve

TRANSLATED FROM: cora_matlab/unitTests/hybridDynamics/nonlinearReset/test_nonlinearReset_resolve.m

Authors:       Mark Wetzlinger
Written:       14-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_nonlinearReset_resolve_01_basic():
    """
    TRANSLATED TEST - Basic nonlinearReset resolve test
    
    Tests resolution of local inputs to states of other components.
    """
    try:
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        
        # define reset function
        def reset_func(x, u):
            return np.array([x[0] + u[0], x[1] + u[1]]) if len(x) >= 2 and len(u) >= 2 else x
        
        reset = NonlinearReset(reset_func)
        
        # binds and flows
        stateBinds = [[1, 2, 3], [4, 5]]  # Python list of lists
        inputBinds = [np.array([[2, 1], [0, 1]]), np.array([[1, 1]])]  # Python list of arrays
        sys1 = LinearSys('sys1', np.eye(3), np.zeros((3, 2)), np.zeros((3, 1)), np.array([1, 2, 3]), np.array([0, 100]), 5)
        sys2 = LinearSys('sys2', np.eye(2), np.array([[0], [0]]), np.array([[0], [0]]), np.array([-1, -2]), 0, -10)
        flowList = [sys1, sys2]  # Python list
        
        # resolve input binds
        reset_ = reset.resolve(flowList, stateBinds, inputBinds)
        
        assert reset_ is not None, "Resolved reset should be created"
        assert reset_.preStateDim == reset.preStateDim, "preStateDim should remain"
        assert reset_.postStateDim == reset.postStateDim, "postStateDim should remain"
    except ImportError:
        pytest.skip("NonlinearReset class not yet implemented")

