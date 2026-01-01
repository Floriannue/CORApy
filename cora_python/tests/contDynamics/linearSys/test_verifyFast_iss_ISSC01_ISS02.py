"""
test_verifyFast_iss_ISSC01_ISS02 - iss benchmark from
    the 2023 ARCH competition

Syntax:
    text = test_verifyFast_iss_ISSC01_ISS02()

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
from cora_python.contSet.interval.vertcat import vertcat
from cora_python.contSet.polytope import Polytope
from cora_python.specification.specification import Specification
# verify is a method of linearSys, accessed via sys.verify()


class TestVerifyFastIssISSC01ISS02:
    """Test class for ISS ISSC01-ISS02 benchmark verification"""
    
    def test_verifyFast_iss_ISSC01_ISS02(self):
        """Test ISS benchmark ISSC01-ISS02 from ARCH 2023 competition"""
        # Parameters --------------------------------------------------------------
        
        # MATLAB: R0 = [interval(-0.0001*ones(270,1),0.0001*ones(270,1)); ...
        # MATLAB:       interval([0;0.8;0.9],[0.1;1;1])];
        I1 = Interval(-0.0001 * np.ones((270, 1)), 0.0001 * np.ones((270, 1)))
        I2 = Interval(np.array([[0], [0.8], [0.9]]), np.array([[0.1], [1], [1]]))
        R0 = vertcat(I1, I2)
        
        # MATLAB: params.U = zonotope(0);
        params = {
            'U': Zonotope(np.array([[0]]), np.zeros((1, 0)))
        }
        # MATLAB: params.R0 = zonotope(R0);
        params['R0'] = R0.zonotope()
        # MATLAB: params.tFinal = 20;
        params['tFinal'] = 20
        
        # MATLAB: options = struct();
        # MATLAB: options.verifyAlg = 'reachavoid:supportFunc';
        options = {
            'verifyAlg': 'reachavoid:supportFunc'
        }
        
        # Specifications ----------------------------------------------------------
        
        # forall t: -5e-4 <= y3 <= 5e-4
        # MATLAB: d = 5e-4;
        d = 5e-4
        # MATLAB: P1 = polytope([0 0 1],-d);
        # MATLAB: P2 = polytope([0 0 -1],-d);
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
        
        # construct extended system matrices (inputs as additional states)
        # MATLAB: dim_x = length(A);
        dim_x = len(A)
        # MATLAB: A_  = [A,B;zeros(size(B,2),dim_x + size(B,2))];
        A_ = np.block([[A, B],
                       [np.zeros((B.shape[1], dim_x + B.shape[1]))]])
        # MATLAB: B_  = zeros(dim_x+size(B,2),1);
        B_ = np.zeros((dim_x + B.shape[1], 1))
        # MATLAB: C_  = [C,zeros(size(C,1),size(B,2))];
        C_ = np.block([[C, np.zeros((C.shape[0], B.shape[1]))]])
        
        # construct the linear system object
        # MATLAB: sys = linearSys('iss',A_,B_,[],C_);
        sys = LinearSys('iss', A_, B_, None, C_)
        
        # Verification ------------------------------------------------------------
        
        # min steps needed: 185
        # MATLAB: [res,fals,savedata] = verify(sys,params,options,spec);
        res, fals, savedata = sys.verify(params, options, spec)
        
        # MATLAB: disp("specifications verified: " + res);
        # MATLAB: disp("computation time: " + savedata.tComp);
        print(f"specifications verified: {res}")
        print(f"computation time: {savedata.get('tComp', 'N/A')}")
        
        # Return value ------------------------------------------------------------
        
        # MATLAB: text = ['Spacestation,ISSC01-ISS02,',num2str(res),',',num2str(savedata.tComp)];
        text = f'Spacestation,ISSC01-ISS02,{res},{savedata.get("tComp", "N/A")}'
        
        return text


def test_verifyFast_iss_ISSC01_ISS02():
    """Test function for ISS ISSC01-ISS02 benchmark verification.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestVerifyFastIssISSC01ISS02()
    result = test.test_verifyFast_iss_ISSC01_ISS02()
    
    print(f"test_verifyFast_iss_ISSC01_ISS02: result = {result}")
    return result


if __name__ == "__main__":
    test_verifyFast_iss_ISSC01_ISS02()

