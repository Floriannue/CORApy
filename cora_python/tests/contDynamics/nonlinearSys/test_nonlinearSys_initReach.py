"""
test_nonlinearSys_initReach - unit test function for computing a single
   time interval reachable set for nonlinear dynamics:
   Checks initReach of the nonlinearSys class for the 6 tank example;
   It is checked whether partial reachable sets and the set
   of linearization errors are correctly obtained

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_initReach.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Matthias Althoff
Written:       31-July-2017
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import math
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


class TestNonlinearSysInitReach:
    """Test class for nonlinearSys initReach functionality"""
    
    def test_nonlinearSys_initReach(self):
        """Test initReach for 6D tank example"""
        # model parameters
        dim_x = 6
        params = {
            'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
            'tFinal': 4,
            'uTrans': np.zeros((1, 1))  # Default: center of input set U
        }
        
        # reachability settings
        options = {
            'timeStep': 4,
            'taylorTerms': 4,
            'zonotopeOrder': 50,
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # system dynamics
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        
        # options check
        # MATLAB: [params,options] = validateOptions(tank,params,options,'FunctionName','reach');
        # Note: validateOptions is not yet translated, so maxError will be set by initReach if missing
        
        # compute derivations (explicitly, since reach-function is not called)
        tank.derivatives(options)
        
        # obtain factors for reachability analysis
        options['factor'] = []
        for i in range(1, options['taylorTerms'] + 2):  # MATLAB: 1:(options.taylorTerms+1)
            # compute initial state factor
            options['factor'].append((options['timeStep'] ** i) / math.factorial(i))
        
        # compute only first step
        Rfirst, _ = tank.initReach(params['R0'], params, options)
        
        # obtain interval hull of reachable set of first point in time
        IH_tp = Interval(Rfirst['tp'][0]['set'])
        # obtain interval hull of reachable set of first time interval
        IH_ti = Interval(Rfirst['ti'][0])
        # obtain linearization errors
        linErrors = Rfirst['tp'][0]['error']
        
        # ground truth
        IH_tp_true = Interval(
            np.array([[1.8057949711597598], [3.6433030183959114], [3.7940260617482671], 
                     [1.9519553317477598], [9.3409949650858550], [4.0928655724716370]]),
            np.array([[2.2288356782079028], [4.0572873081850807], [4.1960714210115002], 
                     [2.3451418924166987], [9.7630596270322201], [4.4862797486713282]])
        )
        IH_ti_true = Interval(
            np.array([[1.7699801606999799], [3.6281401930838144], [3.7805292441504390], 
                     [1.7850641948695933], [9.3278848047801457], [3.7900869967590674]]),
            np.array([[2.2489804444442649], [4.2207006906857227], [4.2157311855735484], 
                     [2.3652362932387256], [10.2200546341070346], [4.5042192761237603]])
        )
        linErrors_true = 1e-3 * np.array([[0.206863579523074], [0.314066666873806], 
                                         [0.161658311464827], [0.353255431809860], 
                                         [0.358487021465299], [0.209190642349808]])
        
        # compare results
        # MATLAB I/O pairs from debug_matlab_nonlinearSys_initReach_abstrerr_chain.m
        # Allow small numerical differences in Python vs MATLAB expm/interval arithmetic
        assert IH_tp.isequal(IH_tp_true, 1e-3)
        assert IH_ti.isequal(IH_ti_true, 1e-3)
        assert compareMatrices(linErrors, linErrors_true, 1e-12, "equal", True)


def test_nonlinearSys_initReach():
    """Test function for nonlinearSys initReach method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysInitReach()
    test.test_nonlinearSys_initReach()
    
    print("test_nonlinearSys_initReach: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_initReach()

