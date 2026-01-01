"""
test_nonlinearSys_reach_time - unit test function of nonlinear
   reachability analysis with shifted start time

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_time.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Mark Wetzlinger
Written:       05-June-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_system_dynamics(x, u):
    """
    Test system dynamics function
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        dx: time-derivate of the system state
    """
    dx = np.zeros((3, 1))
    dx[0, 0] = -x[1, 0] * x[2, 0]
    dx[1, 0] = -x[0, 0] + u[0, 0]
    dx[2, 0] = -x[1, 0] * x[0, 0]
    return dx


class TestNonlinearSysReachTime:
    """Test class for nonlinearSys reach functionality with shifted start time"""
    
    def test_nonlinearSys_reach_time(self):
        """Test reach for nonlinear system with shifted start time"""
        # MATLAB: res = true;
        res = True
        
        # MATLAB: n = 3; m = 1;
        n = 3
        m = 1
        # MATLAB: f = @(x,u) [-x(2)*x(3); -x(1) + u(1); -x(2)*x(1)];
        # MATLAB: sys = nonlinearSys(f,n,m);
        sys = NonlinearSys(test_system_dynamics, states=n, inputs=m)
        
        # model parameters
        # MATLAB: params.R0 = zonotope(2*ones(n,1),0.05*diag(ones(n,1)));
        # MATLAB: params.U = zonotope(0,0.01);
        # MATLAB: params.tStart = 0.15;
        # MATLAB: params.tFinal = 0.25;
        params = {
            'R0': Zonotope(2 * np.ones((n, 1)), 0.05 * np.diag(np.ones(n))),
            'U': Zonotope(np.zeros((m, 1)), 0.01 * np.eye(m)),
            'tStart': 0.15,
            'tFinal': 0.25
        }
        
        # call different algorithms
        # MATLAB: options.timeStep = 0.01;
        # MATLAB: options.taylorTerms = 4;
        # MATLAB: options.zonotopeOrder = 20;
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        options = {
            'timeStep': 0.01,
            'taylorTerms': 4,
            'zonotopeOrder': 20,
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # number of steps
        # MATLAB: steps = round((params.tFinal-params.tStart)/options.timeStep);
        steps = round((params['tFinal'] - params['tStart']) / options['timeStep'])
        
        # reachability analysis
        # MATLAB: R = reach(sys,params,options);
        R = reach(sys, params, options)
        
        # check if times are correct
        # MATLAB: assert(withinTol(R.timePoint.time{1},params.tStart))
        assert withinTol(R['timePoint']['time'][0], params['tStart']), \
            f"First time point {R['timePoint']['time'][0]} should equal tStart {params['tStart']}"
        
        # MATLAB: assert(length(R.timePoint.time) == steps + 1)
        assert len(R['timePoint']['time']) == steps + 1, \
            f"Number of time points {len(R['timePoint']['time'])} should equal steps + 1 = {steps + 1}"
        
        # MATLAB: assert(withinTol(R.timePoint.time{end},params.tFinal))
        assert withinTol(R['timePoint']['time'][-1], params['tFinal']), \
            f"Last time point {R['timePoint']['time'][-1]} should equal tFinal {params['tFinal']}"
        
        return res


def test_nonlinearSys_reach_time():
    """Test function for nonlinearSys reach method with shifted start time.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysReachTime()
    result = test.test_nonlinearSys_reach_time()
    
    print("test_nonlinearSys_reach_time: all tests passed")
    return result


if __name__ == "__main__":
    test_nonlinearSys_reach_time()


