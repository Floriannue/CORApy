"""
test_nonlinearSys_reach_02_linearEqualsNonlinear - unit_test_function for 
    nonlinear reachability analysis; it is checked whether the solution
    of a linear system equals the solution obtained by the nonlinear
    system class when the system is linear

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_02_linearEqualsNonlinear.py

Inputs:
    -

Outputs:
    res - true/false 

Authors:       Matthias Althoff
Written:       09-August-2016
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def fiveDimSysEq(x, u):
    """
    fiveDimSysEq - system dynamics for 5D linear system
    
    Args:
        x: state vector
        u: input vector
        
    Returns:
        dx: time-derivate of the system state
    """
    # Linear system: dx = A*x + B*u
    A = np.array([[-1, -4, 0, 0, 0],
                  [4, -1, 0, 0, 0],
                  [0, 0, -3, 1, 0],
                  [0, 0, -1, -3, 0],
                  [0, 0, 0, 0, -2]])
    B = np.array([[1], [1], [1], [1], [1]])
    
    dx = A @ x + B @ u
    return dx


class TestNonlinearSysReach02LinearEqualsNonlinear:
    """Test class for linear equals nonlinear reachability"""
    
    def test_nonlinearSys_reach_02_linearEqualsNonlinear(self):
        """Test that linear and nonlinear systems give same results"""
        # assume true
        # MATLAB: res = true;
        res = True
        
        # Parameters --------------------------------------------------------------
        # MATLAB: dim_x = 5;
        dim_x = 5
        # MATLAB: params.tFinal = 0.04;
        # MATLAB: params.R0 = zonotope(ones(dim_x,1),0.1*eye(dim_x));
        # MATLAB: params.U = zonotope([1; 0; 0; 0.5; -0.5],0.5*diag([0.2, 0.5, 0.2, 0.5, 0.5]));
        params = {
            'tFinal': 0.04,
            'R0': Zonotope(np.ones((dim_x, 1)), 0.1 * np.eye(dim_x)),
            'U': Zonotope(
                np.array([[1], [0], [0], [0.5], [-0.5]]),
                0.5 * np.diag([0.2, 0.5, 0.2, 0.5, 0.5])
            )
        }
        
        # Reachability Settings (linear) ------------------------------------------
        # MATLAB: optionsLin.timeStep = 0.04;
        # MATLAB: optionsLin.taylorTerms = 8;
        # MATLAB: optionsLin.zonotopeOrder = 10;
        # MATLAB: optionsLin.linAlg = 'wrapping-free';
        optionsLin = {
            'timeStep': 0.04,
            'taylorTerms': 8,
            'zonotopeOrder': 10,
            'linAlg': 'wrapping-free'
        }
        
        # Reachability Settings (nonlinear) ---------------------------------------
        # MATLAB: optionsNonLin.timeStep = optionsLin.timeStep;
        # MATLAB: optionsNonLin.taylorTerms = optionsLin.taylorTerms;
        # MATLAB: optionsNonLin.zonotopeOrder = optionsLin.zonotopeOrder;
        optionsNonLin = {
            'timeStep': optionsLin['timeStep'],
            'taylorTerms': optionsLin['taylorTerms'],
            'zonotopeOrder': optionsLin['zonotopeOrder']
        }
        
        # System Dynamics ---------------------------------------------------------
        # linear system
        # MATLAB: A = [-1 -4 0 0 0; 4 -1 0 0 0; 0 0 -3 1 0; 0 0 -1 -3 0; 0 0 0 0 -2]; B = 1;
        # MATLAB: fiveDimSys = linearSys('fiveDimSys',A,B);
        A = np.array([[-1, -4, 0, 0, 0],
                      [4, -1, 0, 0, 0],
                      [0, 0, -3, 1, 0],
                      [0, 0, -1, -3, 0],
                      [0, 0, 0, 0, -2]])
        B = np.array([[1], [1], [1], [1], [1]])
        fiveDimSys = LinearSys(A, B, name='fiveDimSys')
        
        # nonlinear system
        # MATLAB: fiveDimSysNonlinear = nonlinearSys(@fiveDimSysEq);
        fiveDimSysNonlinear = NonlinearSys(fiveDimSysEq, states=5, inputs=1)
        
        # Reachability Analysis ---------------------------------------------------
        # linear system
        # MATLAB: Rlin = reach(fiveDimSys, params, optionsLin);
        from cora_python.contDynamics.contDynamics.reach import reach
        Rlin = reach(fiveDimSys, params, optionsLin)
        
        # nonlinear system (conservative linearization)
        # MATLAB: optionsNonLin.alg = 'lin';
        # MATLAB: optionsNonLin.tensorOrder = 2;
        # MATLAB: Rnonlin1 = reach(fiveDimSysNonlinear, params, optionsNonLin);
        optionsNonLin1 = optionsNonLin.copy()
        optionsNonLin1['alg'] = 'lin'
        optionsNonLin1['tensorOrder'] = 2
        Rnonlin1 = reach(fiveDimSysNonlinear, params, optionsNonLin1)
        
        # nonlinear system (conservative polynomialization)
        # MATLAB: optionsNonLin.alg = 'poly';
        # MATLAB: optionsNonLin.tensorOrder = 3;
        # MATLAB: optionsNonLin.intermediateOrder = 100*optionsNonLin.zonotopeOrder;
        # MATLAB: optionsNonLin.errorOrder = 1;
        # MATLAB: Rnonlin2 = reach(fiveDimSysNonlinear, params, optionsNonLin);
        optionsNonLin2 = optionsNonLin.copy()
        optionsNonLin2['alg'] = 'poly'
        optionsNonLin2['tensorOrder'] = 3
        optionsNonLin2['intermediateOrder'] = 100 * optionsNonLin['zonotopeOrder']
        optionsNonLin2['errorOrder'] = 1
        Rnonlin2 = reach(fiveDimSysNonlinear, params, optionsNonLin2)
        
        # Numerical Evaluation ----------------------------------------------------
        # enclosing intervals of final reachable sets
        # MATLAB: IH = interval(Rlin.timePoint.set{end});
        # MATLAB: IH_nonlinear_T1 = interval(Rnonlin1.timePoint.set{end});
        # MATLAB: IH_nonlinear_T2 = interval(Rnonlin2.timePoint.set{end});
        IH = Rlin['timePoint']['set'][-1].interval()
        IH_nonlinear_T1 = Rnonlin1['timePoint']['set'][-1].interval()
        IH_nonlinear_T2 = Rnonlin2['timePoint']['set'][-1].interval()
        
        # final result
        # MATLAB: assert(isequal(IH,IH_nonlinear_T1,1e-8) && isequal(IH,IH_nonlinear_T2,1e-8));
        assert Interval.isequal(IH, IH_nonlinear_T1, 1e-8), \
            "Linear and nonlinear (lin) results should match"
        assert Interval.isequal(IH, IH_nonlinear_T2, 1e-8), \
            "Linear and nonlinear (poly) results should match"


def test_nonlinearSys_reach_02_linearEqualsNonlinear():
    """Test function for linear equals nonlinear reachability.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysReach02LinearEqualsNonlinear()
    test.test_nonlinearSys_reach_02_linearEqualsNonlinear()
    
    print("test_nonlinearSys_reach_02_linearEqualsNonlinear: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_reach_02_linearEqualsNonlinear()

