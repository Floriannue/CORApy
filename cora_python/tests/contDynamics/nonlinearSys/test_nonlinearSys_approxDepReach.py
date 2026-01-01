"""
test_nonlinearSys_approxDepReach - unit test function for approximative 
reachability analysis of nonlinear systems (required for controller 
synthesis)

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_approxDepReach.py

Inputs:
    -

Outputs:
    res - true/false 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Lukas Schäfer
Written:       07-February-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


class TestNonlinearSysApproxDepReach:
    """Test class for approximative reachability analysis"""
    
    def test_nonlinearSys_approxDepReach(self):
        """Test approximative vs over-approximative reachability analysis"""
        # dynamical system
        # MATLAB: tank = nonlinearSys(@tank6Eq);
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        
        # parameters for reachability analysis
        # MATLAB: dim_x=6;
        dim_x = 6
        # MATLAB: params.tFinal = 1; %final time
        # MATLAB: params.R0 = polyZonotope([2; 4; 4; 2; 10; 4],0.2*eye(dim_x));
        # MATLAB: params.U = zonotope([0,0.005]);
        params = {
            'tFinal': 1,  # final time
            'R0': PolyZonotope(
                np.array([[2], [4], [4], [2], [10], [4]]),
                0.2 * np.eye(dim_x),
                np.array([]).reshape(dim_x, 0),  # Empty GI
                np.array([]).reshape(0, 0)  # Empty E
            ),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1))
        }
        
        # algorithm settings for reachability analysis - 
        # MATLAB: options.zonotopeOrder = 1000; % choose a large value to avoid order reduction of the time-point reachable set
        # MATLAB: options.taylorTerms = 5;
        # MATLAB: options.tensorOrder = 3;
        # MATLAB: options.intermediateOrder = 10;
        # MATLAB: options.errorOrder = 10;
        # MATLAB: options.reductionTechnique = 'girard';
        # MATLAB: options.alg = 'poly';
        # MATLAB: options.timeStep = params.tFinal; % only one time step
        options = {
            'zonotopeOrder': 1000,  # choose a large value to avoid order reduction
            'taylorTerms': 5,
            'tensorOrder': 3,
            'intermediateOrder': 10,
            'errorOrder': 10,
            'reductionTechnique': 'girard',
            'alg': 'poly',
            'timeStep': params['tFinal']  # only one time step
        }
        
        # compute over-approximation of the reachable set
        # MATLAB: options.approxDepOnly = false;
        # MATLAB: R = reach(tank,params,options);
        options_over = options.copy()
        options_over['approxDepOnly'] = False
        from cora_python.contDynamics.contDynamics.reach import reach
        R = reach(tank, params, options_over)
        
        # compute approximation of the reachable set
        # MATLAB: options.approxDepOnly = true;
        # MATLAB: Rapprox = reach(tank,params,options);
        options_approx = options.copy()
        options_approx['approxDepOnly'] = True
        Rapprox = reach(tank, params, options_approx)
        
        # -> since the same parameters are used and we chose a large zonotope
        # order (i.e., we do not need to account for reduction errors), we expect
        # the approximative time-point reachable set to be a subset of the
        # over-approximative reachable set - the same should apply for the
        # respective zonotope enclosures since the additional generators of the
        # zonotope representing the over-approximative reachable set are stored in 
        # the independent generator matrix
        # MATLAB: assert(contains(zonotope(R.timePoint.set{end}),zonotope(Rapprox.timePoint.set{end}),'approx:st',1e-6));
        from cora_python.contSet.zonotope import Zonotope
        from cora_python.contSet.zonotope.contains import contains
        
        R_end_zono = Zonotope(R['timePoint']['set'][-1])
        Rapprox_end_zono = Zonotope(Rapprox['timePoint']['set'][-1])
        
        # Check that Rapprox is contained in R
        assert contains(R_end_zono, Rapprox_end_zono, 'approx:st', 1e-6), \
            "Approximative reachable set should be subset of over-approximative set"
        
        # combine results
        # MATLAB: res = true;
        res = True
        assert res


def test_nonlinearSys_approxDepReach():
    """Test function for approximative reachability analysis.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysApproxDepReach()
    test.test_nonlinearSys_approxDepReach()
    
    print("test_nonlinearSys_approxDepReach: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_approxDepReach()

