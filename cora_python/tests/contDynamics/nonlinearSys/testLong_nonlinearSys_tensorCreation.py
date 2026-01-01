"""
testLong_nonlinearSys_tensorCreation - unit_test_function for the
   creation of the third-order-tensor file

   Checks different scenarios of settings, where each scenario results in
   a different third-order tensor

Syntax:
    res = testLong_nonlinearSys_tensorCreation()

Inputs:
    -

Outputs:
    res - true/false

Authors:       Niklas Kochdumper
Written:       02-August-2018
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


class TestLongNonlinearSysTensorCreation:
    """Test class for nonlinearSys tensor creation functionality"""
    
    def test_long_nonlinearSys_tensorCreation(self):
        """Test creation of third-order tensor files with different option combinations"""
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 8; % final time
        # MATLAB: params.R0 = zonotope([[2; 4; 4; 2; 10; 4],0.2*eye(6)]); % initial set
        # MATLAB: params.U = zonotope([0,0.005]); %input for reachability analysis
        params = {
            'tFinal': 8,  # final time
            'R0': Zonotope(
                np.array([[2], [4], [4], [2], [10], [4]]),
                0.2 * np.eye(6)
            ),
            'U': Zonotope(np.array([[0]]), np.array([[0.005]]))
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep = 4;
        # MATLAB: options.taylorTerms = 4;
        # MATLAB: options.zonotopeOrder = 50;
        # MATLAB: options.intermediateOrder = 5;
        # MATLAB: options.errorOrder = 1;
        options = {
            'timeStep': 4,
            'taylorTerms': 4,
            'zonotopeOrder': 50,
            'intermediateOrder': 5,
            'errorOrder': 1
        }
        
        # Test Cases --------------------------------------------------------------
        # For each of the considered scenarios, a different third-order-tensor file
        # is created
        
        # MATLAB: for i = 1:2
        for i in range(1, 3):
            # MATLAB: for j = 1:2
            for j in range(1, 3):
                # MATLAB: for k = 1:2
                for k in range(1, 3):
                    # MATLAB: for h = 1:2
                    for h in range(1, 3):
                        # MATLAB: for m = 1:2
                        for m in range(1, 3):
                            
                            # MATLAB: options_ = options;
                            options_ = options.copy()
                            
                            # Initialize lagrangeRem if not present
                            if 'lagrangeRem' not in options_:
                                options_['lagrangeRem'] = {}
                            
                            # replacements
                            # MATLAB: if i == 1
                            if i == 1:
                                # MATLAB: options_.lagrangeRem.replacements = @(x,u) 897680497035489/(36028797018963968*x(5)^(5/2));
                                def replacements_func(x, u):
                                    return 897680497035489 / (36028797018963968 * x[4, 0]**(5/2))
                                options_['lagrangeRem']['replacements'] = replacements_func
                            
                            # parallel execution
                            # MATLAB: if j == 1
                            if j == 1:
                                # MATLAB: options_.lagrangeRem.tensorParallel = true;
                                options_['lagrangeRem']['tensorParallel'] = True
                            
                            # reachability algorithm
                            # MATLAB: if k == 1
                            if k == 1:
                                # MATLAB: options_.alg = 'poly';
                                options_['alg'] = 'poly'
                            else:
                                # MATLAB: options_.alg = 'lin';
                                options_['alg'] = 'lin'
                            
                            # taylor models
                            # MATLAB: if h == 1
                            if h == 1:
                                # MATLAB: options_.lagrangeRem.method = 'taylorModel';
                                options_['lagrangeRem']['method'] = 'taylorModel'
                            
                            # tensor order
                            # MATLAB: if m == 1
                            if m == 1:
                                # MATLAB: if strcmp(options_.alg,'poly')
                                if options_['alg'] == 'poly':
                                    # MATLAB: % tensorOrder = 2 not valid for poly -> skip
                                    # MATLAB: continue
                                    continue
                                # MATLAB: options_.tensorOrder = 2;
                                options_['tensorOrder'] = 2
                            else:
                                # MATLAB: options_.tensorOrder = 3;
                                options_['tensorOrder'] = 3
                            
                            # create system
                            # MATLAB: sys = nonlinearSys(@tank6Eq);
                            sys = NonlinearSys(tank6Eq, states=6, inputs=1)
                            
                            # compute reachable set
                            # MATLAB: reach(sys, params, options_);
                            reach(sys, params, options_)
        
        # test is successful if no error occurred during execution
        # MATLAB: res = true;
        res = True
        
        return res


def testLong_nonlinearSys_tensorCreation():
    """Test function for nonlinearSys tensor creation.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysTensorCreation()
    result = test.test_long_nonlinearSys_tensorCreation()
    
    print("testLong_nonlinearSys_tensorCreation: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_tensorCreation()

