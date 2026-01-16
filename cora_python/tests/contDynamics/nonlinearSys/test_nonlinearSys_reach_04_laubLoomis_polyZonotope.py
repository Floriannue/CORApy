"""
test_nonlinearSys_reach_04_laubLoomis_polyZonotope - unit_test_function of 
   nonlinear reachability analysis: Checks the solution of a 7D nonlinear
   example using a non-convex set representation;

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_04_laubLoomis_polyZonotope.py

Inputs:
    -

Outputs:
    res - true/false 

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       26-January-2016
Last update:   23-April-2020 (restructure params/options)
               22-June-2020 (NK, adapted to polyZonotope set rep.)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.interval import Interval
from cora_python.models.Cora.laubLoomis import laubLoomis




class TestNonlinearSysReach04LaubLoomisPolyZonotope:
    """Test class for nonlinearSys reach functionality (Laub-Loomis with polyZonotope)"""
    
    def test_nonlinearSys_reach_04_laubLoomis_polyZonotope(self):
        """Test reach for 7D Laub-Loomis example using polyZonotope"""
        # assume true
        # MATLAB: res = true;
        res = True
        
        # Parameters --------------------------------------------------------------
        # MATLAB: x0 = [1.2; 1.05; 1.5; 2.4; 1; 0.1; 0.45];
        x0 = np.array([[1.2], [1.05], [1.5], [2.4], [1], [0.1], [0.45]])
        # MATLAB: R0 = zonotope([x0,0.3*eye(7)]);
        R0 = Zonotope(x0, 0.3 * np.eye(7))
        # MATLAB: params.R0 = polyZonotope(R0);                          % initial set
        params = {
            'R0': PolyZonotope(R0),  # initial set
            'tFinal': 0.2  # final time
        }
        
        # Reachability Settings ---------------------------------------------------
        # settings
        # MATLAB: options.timeStep = 0.01;
        # MATLAB: options.taylorTerms = 20;
        # MATLAB: options.zonotopeOrder = 100;
        # MATLAB: options.intermediateOrder = 50;
        # MATLAB: options.errorOrder = 15;
        # MATLAB: options.alg = 'poly';
        # MATLAB: options.tensorOrder = 3;
        options = {
            'timeStep': 0.01,
            'taylorTerms': 20,
            'zonotopeOrder': 100,
            'intermediateOrder': 50,
            'errorOrder': 15,
            'alg': 'poly',
            'tensorOrder': 3
        }
        
        # settings for polynomial zonotopes
        # MATLAB: polyZono.maxDepGenOrder = 30;
        # MATLAB: polyZono.maxPolyZonoRatio = 0.001;
        # MATLAB: polyZono.restructureTechnique = 'reduceFullGirard';
        # MATLAB: options.polyZono = polyZono;
        polyZono = {
            'maxDepGenOrder': 30,
            'maxPolyZonoRatio': 0.001,
            'restructureTechnique': 'reduceFullGirard'
        }
        options['polyZono'] = polyZono
        
        # System Dynamics ---------------------------------------------------------
        # MATLAB: sys = nonlinearSys(@laubLoomis);
        sys = NonlinearSys(laubLoomis)
        
        # Reachability Analysis ---------------------------------------------------
        # MATLAB: R = reach(sys, params, options);
        R = sys.reach(params, options)
        
        # Numerical Evaluation ----------------------------------------------------
        # enclose result by interval
        # MATLAB: IH = interval(R.timeInterval.set{end});
        IH = Interval(R.timeInterval.set[-1])
        
        # saved result
        # MATLAB: IH_saved = interval( ...
        #     [1.0149071383850252; 0.8008249447366792; 0.9260067912066403; 1.6096886042592162; 0.4931654608095589; -0.0696709952280837; 0.0041738752897748], ...
        #     [1.6994210307009445; 1.5100585708305747; 1.6826517863619266; 2.4182337203368247; 1.1023506577076780; 0.2924850920995423; 0.7099585755176793]);
        IH_saved = Interval(
            np.array([[1.0149071383850252], [0.8008249447366792], [0.9260067912066403], 
                     [1.6096886042592162], [0.4931654608095589], [-0.0696709952280837], [0.0041738752897748]]),
            np.array([[1.6994210307009445], [1.5100585708305747], [1.6826517863619266], 
                     [2.4182337203368247], [1.1023506577076780], [0.2924850920995423], [0.7099585755176793]])
        )
        
        # final result
        # MATLAB: assert(isequal(IH,IH_saved,1e-8));
        assert IH.isequal(IH_saved, 1e-8), \
            f"Interval hull mismatch: IH={IH}, IH_saved={IH_saved}"


def test_nonlinearSys_reach_04_laubLoomis_polyZonotope():
    """Test function for nonlinearSys reach method (Laub-Loomis with polyZonotope).
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysReach04LaubLoomisPolyZonotope()
    test.test_nonlinearSys_reach_04_laubLoomis_polyZonotope()
    
    print("test_nonlinearSys_reach_04_laubLoomis_polyZonotope: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_reach_04_laubLoomis_polyZonotope()

