"""
test_verifyFast_iss_ISSF01_ISU01 - iss benchmark from
    the 2023 ARCH competition

Syntax:
    text = test_verifyFast_iss_ISSF01_ISU01()

Inputs:
    -

Outputs:
    text - string

Authors:       Mark Wetzlinger
Written:       23-March-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import scipy.io
import os
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope import Polytope
from cora_python.specification.specification import Specification
# verify is a method of linearSys, accessed via sys.verify()


class TestVerifyFastIssISSF01ISU01:
    """Test class for ISS ISSF01-ISU01 benchmark verification"""
    
    def test_verifyFast_iss_ISSF01_ISU01(self):
        """Test ISS benchmark ISSF01-ISU01 from ARCH 2023 competition"""
        # Parameters --------------------------------------------------------------
        
        # initial set
        # MATLAB: R0 = interval(-0.0001*ones(270,1),0.0001*ones(270,1));
        R0 = Interval(-0.0001 * np.ones((270, 1)), 0.0001 * np.ones((270, 1)))
        # MATLAB: params.R0 = zonotope(R0);
        params = {
            'R0': R0.zonotope()
        }
        
        # uncertain inputs
        # MATLAB: U = interval([0;0.8;0.9],[0.1;1;1]);
        U = Interval(np.array([[0], [0.8], [0.9]]), np.array([[0.1], [1], [1]]))
        # MATLAB: params.U = zonotope(U);
        params['U'] = U.zonotope()
        
        # final time
        # MATLAB: params.tFinal = 20;
        params['tFinal'] = 20
        
        # MATLAB: options = struct();
        # MATLAB: options.verifyAlg = 'reachavoid:supportFunc';
        options = {
            'verifyAlg': 'reachavoid:supportFunc'
        }
        
        # Specification -----------------------------------------------------------
        
        # forall t: -5e-4 <= y3 <= 5e-4
        # MATLAB: d = 5e-4;
        d = 5e-4
        # MATLAB: P1 = polytope([0, 0, 1], -d);
        # MATLAB: P2 = polytope([0, 0, -1], -d);
        P1 = Polytope(np.array([[0, 0, 1]]), np.array([[-d]]))
        P2 = Polytope(np.array([[0, 0, -1]]), np.array([[-d]]))
        # MATLAB: spec = specification({P1,P2},'unsafeSet');
        spec = Specification([P1, P2], 'unsafeSet')
        
        # System Dynamics ---------------------------------------------------------
        
        # load system matrices
        # MATLAB: load iss.mat A B C
        from cora_python.g.macros.CORAROOT import CORAROOT
        mat_file_path = os.path.join(CORAROOT(), 'models', 'Cora', 'iss.mat')
        data = scipy.io.loadmat(mat_file_path)
        A = data['A']
        B = data['B']
        C = data['C']
        
        # construct the linear system object
        # MATLAB: sys = linearSys('iss',A,B,[],C);
        sys = LinearSys('iss', A, B, None, C)
        
        # Verification ------------------------------------------------------------
        
        # min steps needed: 150
        # MATLAB: [res,fals,savedata] = verify(sys,params,options,spec);
        res, fals, savedata = sys.verify(params, options, spec)
        
        # MATLAB: disp("specifications verified: " + res);
        # MATLAB: disp("computation time: " + savedata.tComp);
        print(f"specifications verified: {res}")
        print(f"computation time: {savedata.get('tComp', 'N/A')}")
        
        # Verify the result
        # MATLAB expects: res = False (unsafeSet reached, specification violated)
        assert res is not None, "Verification result should not be None"
        assert res == False, f"Specification should be violated (unsafeSet reached), got {res}"
        assert savedata.get('tComp', 0) > 0, "Computation time should be positive"


def test_verifyFast_iss_ISSF01_ISU01():
    """Test function for ISS ISSF01-ISU01 benchmark verification.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestVerifyFastIssISSF01ISU01()
    test.test_verifyFast_iss_ISSF01_ISU01()


if __name__ == "__main__":
    test_verifyFast_iss_ISSF01_ISU01()

