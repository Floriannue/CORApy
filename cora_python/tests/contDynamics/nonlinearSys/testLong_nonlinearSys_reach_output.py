"""
testLong_nonlinearSys_reach_output - tests if output equation works

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_output.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       19-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.contDynamics.reach import reach


def f(x, u):
    """
    Dynamic equation for the test system
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        dx: time-derivative of the system state
    """
    dx = np.zeros((2, 1))
    dx[0, 0] = x[0, 0]**2 - x[1, 0]
    dx[1, 0] = x[1, 0] - u[0, 0]
    return dx


def g_lin(x, u):
    """
    Linear output equation
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        y: output
    """
    return np.array([[x[0, 0] + x[1, 0]]])


def g_quad(x, u):
    """
    Quadratic output equation
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        y: output
    """
    return np.array([[x[0, 0] + x[1, 0]**2]])


def g_cub(x, u):
    """
    Cubic output equation
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        y: output
    """
    return np.array([[x[0, 0] + x[1, 0]**3]])


class TestLongNonlinearSysReachOutput:
    """Test class for nonlinearSys reach functionality with output equations"""
    
    def test_long_nonlinearSys_reach_output(self):
        """Test reach for nonlinearSys with different output equations"""
        # assume satisfaction
        # MATLAB: res = true;
        res = True
        
        # model parameters
        # MATLAB: params.tFinal = 0.1;
        # MATLAB: params.R0 = zonotope(ones(2,1),0.05*eye(2));
        # MATLAB: params.U = zonotope(0,0.01);
        params = {
            'tFinal': 0.1,
            'R0': Zonotope(np.ones((2, 1)), 0.05 * np.eye(2)),
            'U': Zonotope(np.zeros((1, 1)), 0.01 * np.eye(1))
        }
        
        # reachability settings
        # MATLAB: options.timeStep = 0.01;
        # MATLAB: options.taylorTerms = 4;
        # MATLAB: options.zonotopeOrder = 30;
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        # MATLAB: options.tensorOrderOutput = 2;
        options = {
            'timeStep': 0.01,
            'taylorTerms': 4,
            'zonotopeOrder': 30,
            'alg': 'lin',
            'tensorOrder': 2,
            'tensorOrderOutput': 2
        }
        
        # dynamic equation
        # MATLAB: f = @(x,u) [x(1)^2 - x(2); x(2) - u(1)];
        # (defined above as function f)
        
        # linear output equation
        # MATLAB: g_lin = @(x,u) x(1) + x(2);
        # (defined above as function g_lin)
        
        # quadratic output equation
        # MATLAB: g_quad = @(x,u) x(1) + x(2)^2;
        # (defined above as function g_quad)
        
        # cubic output equation
        # MATLAB: g_cub = @(x,u) x(1) + x(2)^3;
        # (defined above as function g_cub)
        
        # instantiate nonlinearSys objects
        # MATLAB: sys1 = nonlinearSys('sys1',f,g_lin);
        # MATLAB: sys2 = nonlinearSys('sys2',f,g_quad);
        # MATLAB: sys3 = nonlinearSys('sys3',f,g_cub);
        sys1 = NonlinearSys(f, states=2, inputs=1, out_fun=g_lin, outputs=1, name='sys1')
        sys2 = NonlinearSys(f, states=2, inputs=1, out_fun=g_quad, outputs=1, name='sys2')
        sys3 = NonlinearSys(f, states=2, inputs=1, out_fun=g_cub, outputs=1, name='sys3')
        
        # reachability analysis
        # MATLAB: reach(sys1,params,options);
        # MATLAB: reach(sys2,params,options);
        # MATLAB: options.tensorOrderOutput = 3;
        # MATLAB: reach(sys3,params,options);
        R1 = reach(sys1, params, options)
        R2 = reach(sys2, params, options)
        options['tensorOrderOutput'] = 3
        R3 = reach(sys3, params, options)
        
        # Verify that all reachability analyses completed successfully
        assert R1 is not None, "Reachability analysis with linear output should complete"
        assert R2 is not None, "Reachability analysis with quadratic output should complete"
        assert R3 is not None, "Reachability analysis with cubic output should complete"
        
        return res


def testLong_nonlinearSys_reach_output():
    """Test function for nonlinearSys reach method with output equations.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReachOutput()
    result = test.test_long_nonlinearSys_reach_output()
    
    print("testLong_nonlinearSys_reach_output: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reach_output()

