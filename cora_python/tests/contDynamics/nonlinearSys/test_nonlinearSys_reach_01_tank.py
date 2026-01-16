"""
test_nonlinearSys_reach_01_tank - unit_test_function of nonlinear
   reachability analysis; Checks the solution of the nonlinearSys class
   for the 6 tank example from [1]; It is checked whether the enclosing 
   interval of the final reachable set is close to an interval provided
   by a previous solution that has been saved

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_01_tank.py

Inputs:
    -

Outputs:
    res - true/false

Reference:
    [1] M. Althoff, O. Stursberg, and M. Buss: Reachability Analysis of 
        Nonlinear Systems with Uncertain Parameters using Conservative 
        Linearization. Proc. of the 47th IEEE Conference on Decision and 
        Control, 2008.

Authors:       Matthias Althoff
Written:       21-July-2016
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


class TestNonlinearSysReach01Tank:
    """Test class for nonlinearSys reach functionality (tank example)"""
    
    def test_nonlinearSys_reach_01_tank(self):
        """Test reach for 6D tank example"""
        # assume true
        # MATLAB: res = true;
        res = True
        
        # Parameters --------------------------------------------------------------
        # MATLAB: dim_x=6;
        dim_x = 6
        # MATLAB: params.tFinal=400; %final time
        # MATLAB: params.R0=zonotope([[2; 4; 4; 2; 10; 4],0.2*eye(dim_x)]);
        # MATLAB: params.U = zonotope([0,0.005]);
        params = {
            'tFinal': 400,  # final time
            'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1))
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep=4; %time step size for reachable set computation
        # MATLAB: options.taylorTerms=4; %number of taylor terms for reachable sets
        # MATLAB: options.zonotopeOrder=50; %zonotope order
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        options = {
            'timeStep': 4,  # time step size for reachable set computation
            'taylorTerms': 4,  # number of taylor terms for reachable sets
            'zonotopeOrder': 50,  # zonotope order
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # System Dynamics----------------------------------------------------------
        # MATLAB: tank = nonlinearSys(@tank6Eq); %initialize tank system
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: R = reach(tank,params,options);
        # Note: reach function should be in contDynamics.reach or nonlinearSys.reach
        R = tank.reach(params, options)
        
        # Numerical Evaluation ----------------------------------------------------
        # MATLAB: IH = interval(R.timeInterval.set{end});
        IH = Interval(R.timeInterval.set[-1])
        
        # MATLAB: IH_saved = interval( ...
        IH_saved = Interval(
            np.array([[2.2936477632745289], [2.1706800635653218], [2.0156503777163191], 
                     [1.8474995639752518], [1.6412146827755991], [1.2961102589990721]]),
            np.array([[3.8377956209405877], [3.6742657882546377], [3.4340315367183956], 
                     [3.1916520121356604], [3.0614948721719735], [3.2481410926468546]])
        )
        
        # MATLAB: %final result
        # MATLAB: assert(isequal(IH,IH_saved,1e-8));
        assert IH.isequal(IH_saved, 1e-8)


def test_nonlinearSys_reach_01_tank():
    """Test function for nonlinearSys reach method (tank example).
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysReach01Tank()
    test.test_nonlinearSys_reach_01_tank()
    
    print("test_nonlinearSys_reach_01_tank: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_reach_01_tank()

