"""
testLong_nonlinearSys_reach_03_vanDerPol - unit_test_function of nonlinear
   reachability analysis; Checks the solution of the nonlinearSys class
   for the van der Pol example; The settings are identical to [1].
   It is checked whether the reachable set is enclosed in the initial set
   after a certain amount of time.

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_03_vanDerPol.py

Inputs:
    -

Outputs:
    res - true/false 

References:
    [1] M. Althoff, O. Stursberg, M. Buss
        "Reachability analysis of nonlinear systems with uncertain
        parameters using conservative linearization", CDC 2008

Authors:       Matthias Althoff
Written:       26-June-2009
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.polytope import Polytope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.g.classes.reachSet.query import query
from cora_python.models.Cora.vanDerPol.vanderPolEq import vanderPolEq


class TestLongNonlinearSysReach03VanDerPol:
    """Test class for nonlinearSys reach functionality (vanDerPol long test)"""
    
    def test_long_nonlinearSys_reach_03_vanDerPol(self):
        """Test reach for vanDerPol example with containment check"""
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 6.74;
        # MATLAB: Z0{1} = zonotope([1.4; 2.3],[0.05 0; 0 0.05]);
        # MATLAB: params.R0 = zonoBundle(Z0);
        Z0 = [Zonotope(np.array([[1.4], [2.3]]), np.array([[0.05, 0], [0, 0.05]]))]
        params = {
            'tFinal': 6.74,
            'R0': ZonoBundle(Z0)
        }
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options.timeStep = 0.02;
        # MATLAB: options.taylorTerms = 4;
        # MATLAB: options.zonotopeOrder = 10;
        # MATLAB: options.intermediateOrder = 10;
        # MATLAB: options.errorOrder = 5;
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 3;
        # MATLAB: options.maxError = 0.05*[1; 1];
        # MATLAB: options.reductionInterval = 100;
        options = {
            'timeStep': 0.02,
            'taylorTerms': 4,
            'zonotopeOrder': 10,
            'intermediateOrder': 10,
            'errorOrder': 5,
            'alg': 'lin',
            'tensorOrder': 3,
            'maxError': 0.05 * np.ones((2, 1)),
            'reductionInterval': 100
        }
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: vanderPol=nonlinearSys(@vanderPolEq); %initialize van-der-Pol oscillator
        vanderPol = NonlinearSys(vanderPolEq, states=2, inputs=1)
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: R = reach(vanderPol, params, options);
        R = reach(vanderPol, params, options)
        
        # Verification ------------------------------------------------------------
        # obtain array of enclosing polytopes of last reachable set
        # MATLAB: Rfin = query(R,'finalSet');
        Rfin = query(R, 'finalSet')
        
        # Handle both single set and list of sets
        if not isinstance(Rfin, list):
            Rfin = [Rfin]
        
        # MATLAB: Premain = polytope(Rfin{1});
        # MATLAB: for i = 2:length(Rfin)
        # MATLAB:     Premain = Premain | polytope(Rfin{i});
        # MATLAB: end
        from cora_python.contSet.polytope.or_ import or_
        Premain = Polytope(Rfin[0])
        for i in range(1, len(Rfin)):
            Premain = or_(Premain, Polytope(Rfin[i]))
        
        # check containment in hull of first couple of time-interval solutions
        # MATLAB: P_hull = polytope(params.R0);
        P_hull = Polytope(params['R0'])
        # MATLAB: for s=1:6
        # MATLAB:     P_hull = P_hull | polytope(R(1).timeInterval.set{s});
        # MATLAB: end
        # Handle R as list or single object
        if isinstance(R, list):
            R_first = R[0]
        else:
            R_first = R
        
        for s in range(6):  # s from 1 to 6 in MATLAB (0-indexed: 0 to 5)
            P_hull = or_(P_hull, Polytope(R_first['timeInterval']['set'][s]))
        
        # MATLAB: assert(contains(P_hull,Premain));
        from cora_python.contSet.polytope.contains_ import contains_
        res_contains, _, _ = contains_(P_hull, Premain)
        assert res_contains, \
            f"Final reachable set should be contained in hull of first time-interval solutions"
        
        # fix polytope/mldivide for test below...
        # MATLAB: (commented out code for polytope/mldivide)
        # This test is commented out in MATLAB, so we skip it here too
        
        # MATLAB: res = true;
        res = True
        
        return res


def testLong_nonlinearSys_reach_03_vanDerPol():
    """Test function for nonlinearSys reach method (vanDerPol long test).
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReach03VanDerPol()
    result = test.test_long_nonlinearSys_reach_03_vanDerPol()
    
    print("testLong_nonlinearSys_reach_03_vanDerPol: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reach_03_vanDerPol()


