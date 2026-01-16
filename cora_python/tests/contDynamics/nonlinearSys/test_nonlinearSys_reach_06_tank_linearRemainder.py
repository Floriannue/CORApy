"""
test_nonlinearSys_reach_06_tank_linearRemainder - example of
   nonlinear reachability  analysis;
   This example can be found in [1, Sec. 3.4.5] or in [2].

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_06_tank_linearRemainder.py

Inputs:
    -

Outputs:
    res - true/false

References:
    [1] M. Althoff. Reachability analysis and its application to the safety 
        assessment of autonomous cars", Dissertation, TUM 2010.
    [2] M. Althoff, O. Stursberg, and M. Buss. Reachability analysis of
        nonlinear systems with uncertain parameters using conservative
        linearization. In Proc. of the 47th IEEE Conference on
        Decision and Control, pages 4042–4048, 2008.

Authors:       Victor Gassmann
Written:       23-May-2019
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import time
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


class TestNonlinearSysReach06TankLinearRemainder:
    """Test class for nonlinearSys reach functionality (tank with linear remainder)"""
    
    def test_nonlinearSys_reach_06_tank_linearRemainder(self):
        """Test reach for 6D tank example comparing normal and linear remainder algorithms"""
        # MATLAB: dim_x=6;
        dim_x = 6
        # MATLAB: params.tFinal=20; %final time
        # MATLAB: params.R0=zonotope([[2; 4; 4; 2; 10; 4],0.2*eye(dim_x)]);
        # MATLAB: params.U = zonotope([0,0.005]);
        params = {
            'tFinal': 20,  # final time
            'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1))
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep=4; %time step size for reachable set computation
        # MATLAB: options.taylorTerms=4; %number of taylor terms for reachable sets
        # MATLAB: options.zonotopeOrder=50; %zonotope order
        # MATLAB: options.reductionInterval=1e3;
        # MATLAB: options.maxError = ones(dim_x,1);
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        options = {
            'timeStep': 4,  # time step size for reachable set computation
            'taylorTerms': 4,  # number of taylor terms for reachable sets
            'zonotopeOrder': 50,  # zonotope order
            'reductionInterval': 1e3,
            'maxError': np.ones((dim_x, 1)),
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # System Dynamics----------------------------------------------------------
        # MATLAB: tank = nonlinearSys(@tank6Eq); %initialize tank system
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: tx1 = tic;
        # MATLAB: R_wo_linear = reach(tank, params, options); %with normal remainder
        # MATLAB: tComp1 = toc(tx1);
        # MATLAB: disp(['computation time of reachable set with normal lagrange remainder: ',num2str(tComp1)]);
        tx1 = time.time()
        R_wo_linear = tank.reach(params, options)  # with normal remainder
        tComp1 = time.time() - tx1
        print(f'computation time of reachable set with normal lagrange remainder: {tComp1}')
        
        # MATLAB: tx2 = tic;
        # MATLAB: options.alg = 'linRem';
        # MATLAB: options.intermediateOrder=5;
        # MATLAB: R = reach(tank, params, options); %remainder added to system matrices
        # MATLAB: tComp2 = toc(tx2);
        # MATLAB: disp(['computation time of reachable set with remainder added to system matrix: ',num2str(tComp2)]);
        tx2 = time.time()
        options['alg'] = 'linRem'
        options['intermediateOrder'] = 5
        R = tank.reach(params, options)  # remainder added to system matrices
        tComp2 = time.time() - tx2
        print(f'computation time of reachable set with remainder added to system matrix: {tComp2}')
        
        # Simulation --------------------------------------------------------------
        # MATLAB: simRes = simulateRandom(tank, params);
        from cora_python.contDynamics.contDynamics.simulateRandom import simulateRandom
        simRes = simulateRandom(tank, params)
        
        # Visualization -----------------------------------------------------------
        # MATLAB: (commented out visualization code)
        # Visualization code is commented out in MATLAB, so we skip it here too
        
        # example completed
        # MATLAB: res = true;
        res = True
        
        # Verify that both reachability analyses completed successfully
        assert R_wo_linear is not None, "Reachability analysis with normal remainder should complete"
        assert R is not None, "Reachability analysis with linear remainder should complete"
        
        return res


def test_nonlinearSys_reach_06_tank_linearRemainder():
    """Test function for nonlinearSys reach method (tank with linear remainder).
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysReach06TankLinearRemainder()
    result = test.test_nonlinearSys_reach_06_tank_linearRemainder()
    
    print("test_nonlinearSys_reach_06_tank_linearRemainder: all tests passed")
    return result


if __name__ == "__main__":
    test_nonlinearSys_reach_06_tank_linearRemainder()

