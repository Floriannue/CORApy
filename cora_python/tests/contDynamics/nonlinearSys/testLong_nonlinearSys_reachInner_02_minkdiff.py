"""
testLong_nonlinearSys_reachInner_02_minkdiff - example for the
   computation of an inner approximation of the reachable set for
   nonlinear dynamics, using the Minkowski difference approach from [1]

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reachInner_02_minkdiff.py

Inputs:
    -

Outputs:
    res - true/false 

References:
    [1] M. Wetzlinger, A. Kulmburg, and M. Althoff. "Inner approximations
        of reachable sets for nonlinear systems using the Minkowski
        difference.", IEEE Control Systems Letters, 2024.

Authors:       Mark Wetzlinger
Written:       17-December-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contDynamics.contDynamics.reach import reach
from cora_python.contDynamics.contDynamics.reachInner import reachInner
from cora_python.g.classes.reachSet.query import query
from cora_python.models.Cora.jetEngine import jetEngine


class TestLongNonlinearSysReachInner02Minkdiff:
    """Test class for nonlinearSys reachInner functionality with Minkowski difference"""
    
    def test_long_nonlinearSys_reachInner_02_minkdiff(self):
        """Test inner-approximation using Minkowski difference approach"""
        # System Dynamics ---------------------------------------------------------
        # MATLAB: sys = nonlinearSys('jetEngine',@jetEngine);
        sys = NonlinearSys(jetEngine, states=2, inputs=1, name='jetEngine')
        
        # Parameters --------------------------------------------------------------
        # MATLAB: params.tFinal = 4;
        # MATLAB: R0 = interval([0.9;0.9],[1.1;1.1]);
        R0 = Interval(np.array([[0.9], [0.9]]), np.array([[1.1], [1.1]]))
        
        # Reachability Settings ---------------------------------------------------
        # MATLAB: options_outer.alg = 'lin-adaptive';
        options_outer = {
            'alg': 'lin-adaptive'
        }
        
        # MATLAB: options_inner.algInner = 'minkdiff';
        # MATLAB: options_inner.timeStep = 0.01;
        # MATLAB: options_inner.tensorOrder = 2;
        # MATLAB: options_inner.compOutputSet = false;
        options_inner = {
            'algInner': 'minkdiff',
            'timeStep': 0.01,
            'tensorOrder': 2,
            'compOutputSet': False
        }
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: params.R0 = zonotope(R0);
        # MATLAB: Rout = reach(sys,params,options_outer);
        params = {
            'tFinal': 4,
            'R0': R0.zonotope()  # Using object method zonotope()
        }
        Rout = reach(sys, params, options_outer)
        
        # MATLAB: params.R0 = polytope(R0);
        # MATLAB: Rin = reachInner(sys,params,options_inner);
        params['R0'] = Polytope(R0)  # Using Polytope constructor
        Rin = reachInner(sys, params, options_inner)
        
        # Validation --------------------------------------------------------------
        # check if last inner approximation is contained in outer approximation
        # MATLAB: Rout_final = query(Rout,'finalSet');
        # MATLAB: Rin_final = query(Rin,'finalSet');
        # MATLAB: assert(contains(Rout_final, Rin_final));
        Rout_final = query(Rout, 'finalSet')
        Rin_final = query(Rin, 'finalSet')
        
        # Handle both single set and list of sets
        if not isinstance(Rout_final, list):
            Rout_final = [Rout_final]
        if not isinstance(Rin_final, list):
            Rin_final = [Rin_final]
        
        # Check containment for each inner set in the outer sets
        from cora_python.contSet.contSet.contains_ import contains_
        for rin in Rin_final:
            contained = False
            for rout in Rout_final:
                res_contains, _, _ = contains_(rout, rin)
                if res_contains:
                    contained = True
                    break
            assert contained, f"Inner approximation {rin} should be contained in outer approximation"
        
        # test completed
        res = True
        
        return res


def testLong_nonlinearSys_reachInner_02_minkdiff():
    """Test function for nonlinearSys reachInner method with Minkowski difference.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLongNonlinearSysReachInner02Minkdiff()
    result = test.test_long_nonlinearSys_reachInner_02_minkdiff()
    
    print("testLong_nonlinearSys_reachInner_02_minkdiff: all tests passed")
    return result


if __name__ == "__main__":
    testLong_nonlinearSys_reachInner_02_minkdiff()

