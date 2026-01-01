"""
testLong_nonlinearSys_reach_07_VDP_linearRemainder - example of 
   nonlinear reachability analysis;

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_07_VDP_linearRemainder.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Victor Gassmann
Written:       22-May-2019
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import time
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.models.Cora.vanDerPol.vanderPolEq import vanderPolEq


class TestLongNonlinearSysReach07VDPLinearRemainder:
    """Test class for nonlinearSys reach functionality (vanDerPol with linear remainder)"""
    
    def test_long_nonlinearSys_reach_07_VDP_linearRemainder(self):
        """Test reach for vanDerPol example comparing normal and linear remainder algorithms"""
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 2.5;
        # MATLAB: Z0{1} = zonotope([1.4;2.3],diag([0.3,0.05]));
        # MATLAB: params.R0 = Z0{1};
        # MATLAB: params.U = zonotope(0);
        params = {
            'tFinal': 2.5,
            'R0': Zonotope(np.array([[1.4], [2.3]]), np.diag([0.3, 0.05])),
            'U': Zonotope(np.zeros((1, 1)), np.zeros((1, 1)))
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep = 0.02;
        # MATLAB: options.taylorTerms = 4;
        # MATLAB: options.zonotopeOrder = 10;
        # MATLAB: options.tensorOrder = 2;
        # MATLAB: options.maxError = 0.5*[1; 1];
        # MATLAB: options.reductionInterval = 100;
        options = {
            'timeStep': 0.02,
            'taylorTerms': 4,
            'zonotopeOrder': 10,
            'tensorOrder': 2,
            'maxError': 0.5 * np.ones((2, 1)),
            'reductionInterval': 100
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: vanderPol = nonlinearSys(@vanderPolEq);
        vanderPol = NonlinearSys(vanderPolEq, states=2, inputs=1)
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: options.alg = 'lin';
        # MATLAB: tx1 = tic;
        # MATLAB: R_wo_linear = reach(vanderPol,params,options);
        # MATLAB: tComp1 = toc(tx1);
        # MATLAB: disp(['computation time of reachable set with normal remainder: ',num2str(tComp1)]);
        options['alg'] = 'lin'
        tx1 = time.time()
        R_wo_linear = reach(vanderPol, params, options)
        tComp1 = time.time() - tx1
        print(f'computation time of reachable set with normal remainder: {tComp1}')
        
        # MATLAB: tx2 = tic;
        # MATLAB: options.alg = 'linRem';
        # MATLAB: options.intermediateOrder = 10;
        # MATLAB: R = reach(vanderPol,params,options);
        # MATLAB: tComp2 = toc(tx2);
        # MATLAB: disp(['computation time of reachable set with remainder added to system matrices: ',num2str(tComp2)]);
        tx2 = time.time()
        options['alg'] = 'linRem'
        options['intermediateOrder'] = 10
        R = reach(vanderPol, params, options)
        tComp2 = time.time() - tx2
        print(f'computation time of reachable set with remainder added to system matrices: {tComp2}')
        
        # example completed
        # MATLAB: res = true;
        res = True
        
        # Verify that both reachability analyses completed successfully
        assert R_wo_linear is not None, "Reachability analysis with normal remainder should complete"
        assert R is not None, "Reachability analysis with linear remainder should complete"
        
        return res


def testLong_nonlinearSys_reach_07_VDP_linearRemainder():
    """Test function for nonlinearSys reach method (vanDerPol with linear remainder).
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReach07VDPLinearRemainder()
    result = test.test_long_nonlinearSys_reach_07_VDP_linearRemainder()
    
    print("testLong_nonlinearSys_reach_07_VDP_linearRemainder: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reach_07_VDP_linearRemainder()

