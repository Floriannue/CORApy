"""
testLong_nonlinearSys_reach_poly - unit_test_function of nonlinear
   reachability analysis with the conservative polynomialization approach

Checks if the reachable set contains all simulated points

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_poly.py

Inputs:
    -

Outputs:
    res - true/false 

Authors:       Niklas Kochdumper
Written:       04-August-2020
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope.polyZonotope import polyZonotope as zonotope_to_polyZonotope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.contDynamics.contDynamics.simulateRandom import simulateRandom


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
    dx[0, 0] = x[1, 0] * u[0, 0] + u[0, 0] * u[1, 0]
    dx[1, 0] = (1 - x[0, 0]) * x[1, 0] - x[0, 0] + u[1, 0]**2
    return dx


class TestLongNonlinearSysReachPoly:
    """Test class for nonlinearSys reach functionality with poly algorithm"""
    
    def test_long_nonlinearSys_reach_poly(self):
        """Test reach for nonlinearSys with poly algorithm and containment check"""
        # assume true
        # MATLAB: res = true;
        res = True
        
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 0.01;
        # MATLAB: params.R0 = polyZonotope(zonotope([1.4; 2.3],[0.05 0; 0 0.05]));
        # MATLAB: params.U = zonotope([1;2] + interval([-0.1;-0.1],[0.1;0.1]));
        Z0 = Zonotope(np.array([[1.4], [2.3]]), np.array([[0.05, 0], [0, 0.05]]))
        I_U = Interval(np.array([[-0.1], [-0.1]]), np.array([[0.1], [0.1]]))
        U_center = np.array([[1], [2]]) + I_U.c
        U_zonotope = I_U.zonotope()  # Using object method zonotope()
        # Shift the zonotope center
        params = {
            'tFinal': 0.01,
            'R0': zonotope_to_polyZonotope(Z0),
            'U': Zonotope(U_center, U_zonotope.G)
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep = params.tFinal;
        # MATLAB: options.taylorTerms = 4;
        # MATLAB: options.zonotopeOrder = 10;
        # MATLAB: options.intermediateOrder = 10;
        # MATLAB: options.errorOrder = 5;
        # MATLAB: options.alg = 'poly';
        # MATLAB: options.tensorOrder = 3;
        options = {
            'timeStep': 0.01,
            'taylorTerms': 4,
            'zonotopeOrder': 10,
            'intermediateOrder': 10,
            'errorOrder': 5,
            'alg': 'poly',
            'tensorOrder': 3
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: f = @(x,u) [x(2)*u(1) + u(1)*u(2);
        #                     (1-x(1))*x(2)-x(1) + u(2)^2];
        # MATLAB: sys = nonlinearSys(f);
        sys = NonlinearSys(f, states=2, inputs=2)
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: R = reach(sys, params, options);
        R = reach(sys, params, options)
        
        # Simulation --------------------------------------------------------------
        # use boundary of the initial set as new initial set to get critical points 
        # MATLAB: c = center(zonotope(params.R0));
        # MATLAB: G = generators(zonotope(params.R0));
        Z0_zonotope = Zonotope(params['R0'].c, params['R0'].G)
        c = Z0_zonotope.c  # Using object property .c
        G = Z0_zonotope.G  # Using object property .G
        
        # MATLAB: R0{1} = zonotope(c + G(:,1),G(:,2));
        # MATLAB: R0{2} = zonotope(c - G(:,1),G(:,2));
        # MATLAB: R0{3} = zonotope(c + G(:,2),G(:,1));
        # MATLAB: R0{4} = zonotope(c - G(:,2),G(:,1));
        R0 = [
            Zonotope(c + G[:, 0:1], G[:, 1:2]),
            Zonotope(c - G[:, 0:1], G[:, 1:2]),
            Zonotope(c + G[:, 1:2], G[:, 0:1]),
            Zonotope(c - G[:, 1:2], G[:, 0:1])
        ]
        
        # simulation options
        # MATLAB: simOpt.points = 100;
        # MATLAB: simOpt.fracVert = 4e-4;
        # MATLAB: simOpt.fracInpVert = 0.9;
        # MATLAB: simOpt.nrConstInp = 2;
        simOpt = {
            'points': 100,
            'fracVert': 4e-4,
            'fracInpVert': 0.9,
            'nrConstInp': 2
        }
        
        # simulate the system
        # MATLAB: points = [];
        # MATLAB: for i = 1:length(R0)
        # MATLAB:     params.R0 = R0{i};
        # MATLAB:     simRes = simulateRandom(sys, params, simOpt);
        # MATLAB:     for j = 1:length(simRes)
        # MATLAB:         points = [points, simRes(j).x{1}(end,:)']; 
        # MATLAB:     end
        # MATLAB: end
        points = []
        for i in range(len(R0)):
            params['R0'] = R0[i]
            simRes = simulateRandom(sys, params, simOpt)
            
            for j in range(len(simRes)):
                if isinstance(simRes[j], dict):
                    x_data = simRes[j].get('x', [])
                    if len(x_data) > 0:
                        points.append(x_data[0][-1, :].reshape(-1, 1))
                else:
                    # Handle case where simRes[j] is an object with x attribute
                    if hasattr(simRes[j], 'x') and len(simRes[j].x) > 0:
                        points.append(simRes[j].x[0][-1, :].reshape(-1, 1))
        
        if len(points) > 0:
            points = np.hstack(points)
        else:
            points = np.array([]).reshape(2, 0)
        
        # Verification ------------------------------------------------------------
        # check if all points are located inside the time point reachable set
        # MATLAB: pgon = polygon(R.timePoint.set{end},12);
        # MATLAB: res = all(contains(pgon,points));
        from cora_python.contSet.polyZonotope.polygon import polygon
        
        # Get the last time point set
        if isinstance(R, dict):
            timePoint_set = R.get('timePoint', {}).get('set', [])
        else:
            # R might be a ReachSet object
            timePoint_set = R.timePoint.set if hasattr(R, 'timePoint') else []
        
        if len(timePoint_set) > 0:
            last_set = timePoint_set[-1]
            pgon = polygon(last_set, 12)
            
            # Check containment
            from cora_python.contSet.polygon.contains_ import contains_
            res_contains, _, _ = contains_(pgon, points)
            res = np.all(res_contains) if isinstance(res_contains, np.ndarray) else res_contains
        else:
            res = False
        
        assert res, "All simulated points should be contained in the reachable set"
        
        return res


def testLong_nonlinearSys_reach_poly():
    """Test function for nonlinearSys reach method with poly algorithm.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReachPoly()
    result = test.test_long_nonlinearSys_reach_poly()
    
    print("testLong_nonlinearSys_reach_poly: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reach_poly()

