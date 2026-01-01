"""
testLong_nonlinearSys_linError - test if the linearization error
   for nonlinear systems is computed correctly

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_linError.py

Inputs:
    -

Outputs:
    res - true/false 

Authors:       Niklas Kochdumper
Written:       12-November-2018
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import math
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.nonlinearSys.initReach import initReach
from cora_python.contDynamics.contDynamics.derivatives import derivatives
from cora_python.contSet.contSet.randPoint import randPoint
from cora_python.models.Cora.autonomousCar.highorderBicycleDynamics import highorderBicycleDynamics


class TestLongNonlinearSysLinError:
    """Test class for nonlinearSys linearization error computation"""
    
    def test_long_nonlinearSys_linError(self):
        """Test if linearization error is computed correctly"""
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 0.0021;
        # MATLAB: params.U = zonotope([0;0]);
        params = {
            'tFinal': 0.0021,
            'U': Zonotope(np.zeros((2, 1)), np.zeros((2, 1)))
        }
        
        # MATLAB: x0 = zeros(18,1);
        # MATLAB: x0(4) = 19.9536;
        # MATLAB: x0(5) = 18.6195;
        # MATLAB: x0(8) = 0.7098;
        x0 = np.zeros((18, 1))
        x0[3, 0] = 19.9536  # x0(4) in MATLAB (1-indexed)
        x0[4, 0] = 18.6195  # x0(5) in MATLAB (1-indexed)
        x0[7, 0] = 0.7098  # x0(8) in MATLAB (1-indexed)
        
        # MATLAB: G = [[0 0 0 0.0203 0.0125 0.0123 0 0 0 0; ...
        #        0 0 0 -0.0068 -0.0015 0.0001 0.0071 0 0 0; ...
        #        0 0 0 0.1635 0.1524 -0.0041 0 0 0 0; ...
        #        0.15 0.1443 0.1391 0 0 0 0 0 0 0.06; ...
        #        -0.0016 -0.0045 -0.0072 0 0 0 0 0 0.0796 -0.0037; ...
        #        0 0 0 -0.0103 -0.0022 -0.0118 -0.0089 0.0786 0 0; ...
        #        0 0 0 0.0006 -0.0634 0 0 0 0 0; ...
        #        -7.1991 -0.0594 -0.0117 0 0 0 0 0 0 -0.0099]; eye(10)];
        G_part1 = np.array([
            [0, 0, 0, 0.0203, 0.0125, 0.0123, 0, 0, 0, 0],
            [0, 0, 0, -0.0068, -0.0015, 0.0001, 0.0071, 0, 0, 0],
            [0, 0, 0, 0.1635, 0.1524, -0.0041, 0, 0, 0, 0],
            [0.15, 0.1443, 0.1391, 0, 0, 0, 0, 0, 0, 0.06],
            [-0.0016, -0.0045, -0.0072, 0, 0, 0, 0, 0, 0.0796, -0.0037],
            [0, 0, 0, -0.0103, -0.0022, -0.0118, -0.0089, 0.0786, 0, 0],
            [0, 0, 0, 0.0006, -0.0634, 0, 0, 0, 0, 0],
            [-7.1991, -0.0594, -0.0117, 0, 0, 0, 0, 0, 0, -0.0099]
        ])
        G_part2 = np.eye(10)
        G = np.hstack([G_part1, G_part2])
        
        # MATLAB: params.R0 = zonotope([x0,G]);
        params['R0'] = Zonotope(x0, G)
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep = params.tFinal;
        # MATLAB: options.taylorTerms = 5;
        # MATLAB: options.zonotopeOrder = 50;
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        options = {
            'timeStep': 0.0021,
            'taylorTerms': 5,
            'zonotopeOrder': 50,
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: nlnsys = nonlinearSys(@highorderBicycleDynamics);
        nlnsys = NonlinearSys(highorderBicycleDynamics, states=18, inputs=2)
        
        # Reachability Analysis ---------------------------------------------------
        # options check
        # MATLAB: [params,options] = validateOptions(nlnsys,params,options,'FunctionName','reach');
        from cora_python.contDynamics.contDynamics.validateOptions import validateOptions
        params, options = validateOptions(nlnsys, params, options, 'FunctionName', 'reach')
        
        # compute symbolic derivatives
        # MATLAB: derivatives(nlnsys,options);
        derivatives(nlnsys, options)
        
        # obtain factors for initial state and input solution time step
        # MATLAB: r = options.timeStep;
        # MATLAB: for i = 1:(options.taylorTerms+1)
        # MATLAB:     options.factor(i) = r^(i)/factorial(i);
        # MATLAB: end
        r = options['timeStep']
        options['factor'] = [r**i / math.factorial(i) for i in range(1, options['taylorTerms'] + 2)]
        
        # perform one reachability step
        # MATLAB: [R, options] = initReach(nlnsys, params.R0, params, options);
        R, options = initReach(nlnsys, params['R0'], params, options)
        
        # extract the set of linearization errors
        # MATLAB: err = R.tp{1}.error;
        # MATLAB: linError = zonotope([zeros(length(err),1),diag(err)]);
        if isinstance(R, dict):
            err = R.get('tp', [{}])[0].get('error', np.zeros(18))
        else:
            err = R.tp[0].error if hasattr(R, 'tp') else np.zeros(18)
        
        linError = Zonotope(np.zeros((len(err), 1)), np.diag(err))
        
        # evaluate the linearization error for a set of random points
        # MATLAB: p.u = center(params.U);
        # MATLAB: f0prev = nlnsys.mFile(center(params.R0),p.u);
        # MATLAB: p.x = center(params.R0) + f0prev*0.5*options.timeStep;
        p_u = params['U'].c  # Using object property .c
        f0prev = nlnsys.mFile(params['R0'].c, p_u)  # Using object property .c
        p_x = params['R0'].c + f0prev * 0.5 * options['timeStep']
        
        # MATLAB: f0 = nlnsys.mFile(p.x,p.u);
        # MATLAB: [A,~] = nlnsys.jacobian(p.x,p.u);
        f0 = nlnsys.mFile(p_x, p_u)
        A, _ = nlnsys.jacobian(p_x, p_u)
        
        # MATLAB: N = 10000;
        # MATLAB: points = zeros(length(x0),N);
        # MATLAB: for i = 1:N
        # MATLAB:     p = randPoint(params.R0);
        # MATLAB:     points(:,i) = nlnsys.mFile(p,[0;0]) - A*(p - nlnsys.linError.p.x) - f0;
        # MATLAB: end
        N = 10000
        points = np.zeros((len(x0), N))
        for i in range(N):
            p = randPoint(params['R0'], 1, 'standard')
            points[:, i:i+1] = nlnsys.mFile(p, np.zeros((2, 1))) - A @ (p - nlnsys.linError['p']['x']) - f0
        
        # check if the set of linerization error contains all randomly computed points
        # MATLAB: linError = interval(linError);
        # MATLAB: linError = linError([1 3 5 6]);
        # MATLAB: points = points([1 3 5 6],:);
        # MATLAB: assert(all(contains(linError,points)));
        linError_interval = linError.interval()  # Using object method interval()
        # Select dimensions 1, 3, 5, 6 (0-indexed: 0, 2, 4, 5)
        linError_selected = Interval(
            linError_interval.inf[[0, 2, 4, 5], :],
            linError_interval.sup[[0, 2, 4, 5], :]
        )
        points_selected = points[[0, 2, 4, 5], :]
        
        from cora_python.contSet.interval.contains_ import contains_
        res_contains, _, _ = contains_(linError_selected, points_selected)
        assert np.all(res_contains) if isinstance(res_contains, np.ndarray) else res_contains, \
            "All randomly computed linearization error points should be contained in the computed error set"
        
        # test completed
        res = True
        
        return res


def testLong_nonlinearSys_linError():
    """Test function for nonlinearSys linearization error computation.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysLinError()
    result = test.test_long_nonlinearSys_linError()
    
    print("testLong_nonlinearSys_linError: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_linError()

