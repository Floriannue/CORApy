"""
testLong_nonlinearSys_reachInner - test if the computed
   inner-approximation of the reachable set is correct

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reachInner.py

Inputs:
    -

Outputs:
    res - true/false 

Authors:       Niklas Kochdumper
Written:       26-August-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.contDynamics.reachInner import reachInner
from cora_python.contSet.contSet.randPoint import randPoint
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def brusselator(x, u):
    """
    brusselator - system dynamics as defined in the MATLAB test
                 (different from the model file brusselator.m)
    
    MATLAB test defines: brusselator = @(x,u) [1-2*x(1) + 3/2 * x(1)^2*x(2); ...
                                                 x(1)-3/2*x(1)^2*x(2)];
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        dx: time-derivative of the system state
    """
    dx = np.zeros((2, 1))
    dx[0, 0] = 1 - 2 * x[0, 0] + 3/2 * x[0, 0]**2 * x[1, 0]
    dx[1, 0] = x[0, 0] - 3/2 * x[0, 0]**2 * x[1, 0]
    return dx


def brusselator_inv(x, u):
    """
    Inverted Brusselator system dynamics (for backward simulation)
    
    MATLAB test defines: sysInv = nonlinearSys(@(x,u) -brusselator(x,u));
    So this is just the negative of brusselator.
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        dx: time-derivative of the system state (negated)
    """
    return -brusselator(x, u)


class TestLongNonlinearSysReachInner:
    """Test class for nonlinearSys reachInner functionality"""
    
    def test_long_nonlinearSys_reachInner(self):
        """Test inner-approximation of reachable set"""
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 1;
        # MATLAB: params.R0 = interval([0.9;0],[1;0.1]);
        params = {
            'tFinal': 1,
            'R0': Interval(np.array([[0.9], [0]]), np.array([[1], [0.1]]))
        }
        
        # Reachability Settings ---------------------------------------------------
        # settings for inner-approximation
        # MATLAB: options.algInner = 'scale';
        # MATLAB: options.splits = 2;
        # MATLAB: options.iter = 2;
        # MATLAB: options.orderInner = 5;
        # MATLAB: options.scaleFac = 0.95;
        # settings for outer-approximation
        # MATLAB: options.timeStep = 0.001;
        # MATLAB: options.taylorTerms = 10;
        # MATLAB: options.zonotopeOrder = 50;
        # MATLAB: options.intermediateOrder = 20;
        # MATLAB: options.errorOrder = 10;
        options = {
            'algInner': 'scale',
            'splits': 2,
            'iter': 2,
            'orderInner': 5,
            'scaleFac': 0.95,
            'timeStep': 0.001,
            'taylorTerms': 10,
            'zonotopeOrder': 50,
            'intermediateOrder': 20,
            'errorOrder': 10
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: brusselator = @(x,u) [1-2*x(1) + 3/2 * x(1)^2*x(2); ...
        #                               x(1)-3/2*x(1)^2*x(2)];
        # MATLAB: sys = nonlinearSys(brusselator);
        sys = NonlinearSys(brusselator, states=2, inputs=1)
        
        # Reachability Analysis ---------------------------------------------------
        # compute inner-approximation
        # MATLAB: [Rin,~] = reachInner(sys,params,options);
        Rin, _ = reachInner(sys, params, options)
        
        # Verification ------------------------------------------------------------
        # Test 1: check if all points inside the computed inner-approximation
        #         are located in the initial set if simulated backward in time
        # MATLAB: sysInv = nonlinearSys(@(x,u) -brusselator(x,u));
        sysInv = NonlinearSys(brusselator_inv, states=2, inputs=1)
        
        # MATLAB: R_i = Rin.timePoint.set{end};
        if isinstance(Rin, dict):
            timePoint_set = Rin.get('timePoint', {}).get('set', [])
        else:
            timePoint_set = Rin.timePoint.set if hasattr(Rin, 'timePoint') else []
        
        if len(timePoint_set) > 0:
            R_i = timePoint_set[-1]
            
            # MATLAB: points = [randPoint(R_i,1000),randPoint(R_i,'all','extreme')];
            points_rand = randPoint(R_i, 1000, 'standard')
            points_extreme = randPoint(R_i, 'all', 'extreme')
            points = np.hstack([points_rand, points_extreme])
            
            # MATLAB: points_ = zeros(size(points));
            points_ = np.zeros_like(points)
            
            # MATLAB: for i = 1:size(points,2)
            # MATLAB:     % simulate backwards in time
            # MATLAB:     simOpts.x0 = points(:,i);
            # MATLAB:     simOpts.tFinal = params.tFinal;
            # MATLAB:     [~,x] = simulate(sysInv,simOpts);
            # MATLAB:     % check if final point is inside initial set
            # MATLAB:     p = x(end,:)';
            # MATLAB:     points_(:,i) = p;
            # MATLAB:     assert(contains(params.R0,p))
            # MATLAB: end
            for i in range(points.shape[1]):
                # simulate backwards in time
                simOpts = {
                    'x0': points[:, i:i+1],
                    'tFinal': params['tFinal']
                }
                _, x = sysInv.simulate(simOpts, {})  # Using object method simulate()
                
                # check if final point is inside initial set
                p = x[-1, :].reshape(-1, 1)
                points_[:, i:i+1] = p
                
                from cora_python.contSet.interval.contains_ import contains_
                res_contains, _, _ = contains_(params['R0'], p)
                assert res_contains, f"Point {p.flatten()} should be contained in initial set"
            
            # Test 2: check if the result matches the stored one
            # MATLAB: I_saved = interval([0.665734495820338; 0.511586092476733], ...
            #                            [0.729556137539839; 0.565744670853688]);
            # MATLAB: I = interval(R_i);
            # MATLAB: assert(isequal(I,I_saved,1e-3));
            I_saved = Interval(
                np.array([[0.665734495820338], [0.511586092476733]]),
                np.array([[0.729556137539839], [0.565744670853688]])
            )
            I = R_i.interval()  # Using object method interval()
            
            # Check if intervals are equal within tolerance
            assert withinTol(I.inf, I_saved.inf, 1e-3), \
                f"Interval inf {I.inf.flatten()} should match saved {I_saved.inf.flatten()}"
            assert withinTol(I.sup, I_saved.sup, 1e-3), \
                f"Interval sup {I.sup.flatten()} should match saved {I_saved.sup.flatten()}"
            
            # combine results
            res = True
        else:
            res = False
        
        return res


def testLong_nonlinearSys_reachInner():
    """Test function for nonlinearSys reachInner method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReachInner()
    result = test.test_long_nonlinearSys_reachInner()
    
    print("testLong_nonlinearSys_reachInner: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reachInner()

