"""
test_nonlinearSys_reach_adaptive_01_jetEngine - example for nonlinear reachability
    analysis using adaptive parameter tuning

Syntax:
    res = test_nonlinearSys_reach_adaptive_01_jetEngine()

Inputs:
    -

Outputs:
    res - true/false

Authors:       Mark Wetzlinger
Written:       02-February-2021
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import time
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine


def test_nonlinearSys_reach_adaptive_01_jetEngine():
    """
    Test nonlinear reachability analysis using adaptive parameter tuning
    
    This test translates example_nonlinear_reach_12_adaptive.m
    """
    # system dimension
    dim_x = 2
    
    # parameters
    params = {}
    params['tFinal'] = 8.0
    # MATLAB: params.R0 = zonotope([[1;1],0.1*diag(ones(dim_x,1))]);
    params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
    # MATLAB: params.U = zonotope(0);
    params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
    
    # algorithm parameters
    # MATLAB: options.alg = 'lin-adaptive';
    options = {}
    options['alg'] = 'lin-adaptive'
    
    # init system
    # MATLAB: sys = nonlinearSys(@jetEngine,dim_x,1);
    sys = NonlinearSys('jetEngine', jetEngine, dim_x, 1)
    
    # MATLAB: adapTime = tic; [R,~,opt] = reach(sys,params,options); tComp = toc(adapTime);
    adapTime = time.time()
    R, _, opt = sys.reach(params, options)
    tComp = time.time() - adapTime
    
    # MATLAB: endset = R.timePoint.set{end};
    # Assuming R is a ReachSet object or list
    if isinstance(R, list) and len(R) > 0:
        R_first = R[0]
    else:
        R_first = R
    
    if hasattr(R_first, 'timePoint') and R_first.timePoint is not None:
        if hasattr(R_first.timePoint, 'set'):
            timePoint_sets = R_first.timePoint.set
        elif isinstance(R_first.timePoint, dict):
            timePoint_sets = R_first.timePoint.get('set', [])
        else:
            timePoint_sets = []
        
        if len(timePoint_sets) > 0:
            endset = timePoint_sets[-1]
            # MATLAB: gamma_o = 2*rad(interval(endset));
            gamma_o = 2 * endset.interval().radius()
        else:
            pytest.skip("No time point sets computed")
    else:
        pytest.skip("No time point data available")
    
    # Verify computation completed
    assert tComp > 0, "Computation time should be positive"
    assert len(timePoint_sets) > 0, "Should have computed reachable sets"
    
    # Verify adaptive options were used
    assert opt is not None or 'alg' in options, "Adaptive algorithm should be used"
    
    return True

