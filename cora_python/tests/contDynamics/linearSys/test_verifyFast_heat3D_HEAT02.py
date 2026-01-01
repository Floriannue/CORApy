"""
test_verifyFast_heat3D_HEAT02 - heat3d benchmark from
    the 2023 ARCH competition

Syntax:
    text = test_verifyFast_heat3D_HEAT02()

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


def aux_getInit(A, samples, temp):
    """
    Returns an initial state vector
    
    Args:
        A: system matrix
        samples: number of samples per dimension
        temp: temperature value
        
    Returns:
        x0: initial state vector
    """
    # MATLAB: x0 = zeros(size(A,1),1);
    x0 = np.zeros((A.shape[0], 1))
    
    # MATLAB: if samples ~= 5 && (samples < 10 || mod(samples,10) ~= 0)
    if samples != 5 and (samples < 10 or samples % 10 != 0):
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:specialError',
                       'Init region is not evenly divided by discretization!')
    
    # maximum z point for initial region is 0.2 for 5 samples and 0.1 otherwise
    # MATLAB: if samples >= 10
    if samples >= 10:
        # MATLAB: max_z = (0.1/(1/samples));
        max_z = int(0.1 / (1 / samples))
    else:
        # MATLAB: max_z = (0.2/(1/samples));
        max_z = int(0.2 / (1 / samples))
    
    # for every x, generate offsets in z and y directions based on samples
    # MATLAB: for z = 0:max_z
    for z in range(max_z + 1):
        # MATLAB: zoffset = z * samples * samples;
        zoffset = z * samples * samples
        
        # MATLAB: for y = 0:(0.2/(1/samples))
        for y in range(int(0.2 / (1 / samples)) + 1):
            # MATLAB: yoffset = y * samples;
            yoffset = y * samples
            
            # MATLAB: for x = 0:(0.4/(1/samples))
            for x in range(int(0.4 / (1 / samples)) + 1):
                # MATLAB: index = x + yoffset + zoffset;
                index = x + yoffset + zoffset
                
                # MATLAB: x0(index+1) = temp;
                x0[index, 0] = temp
    
    return x0


class TestVerifyFastHeat3DHEAT02:
    """Test class for heat3D HEAT02 benchmark verification"""
    
    def test_verifyFast_heat3D_HEAT02(self):
        """Test heat3D benchmark HEAT02 from ARCH 2023 competition"""
        # System Dynamics ---------------------------------------------------------
        
        # load system matrices
        # MATLAB: load('heat10');
        from cora_python.g.macros.CORAROOT import CORAROOT
        mat_file_path = os.path.join(CORAROOT(), 'models', 'Cora', 'heat3D', 'heat10.mat')
        data = scipy.io.loadmat(mat_file_path)
        A = data['A']
        B = data['B']
        
        # construct output matrix (center of block)
        # MATLAB: samples = nthroot(size(A,1),3);
        samples = int(np.round(np.power(A.shape[0], 1/3)))
        # MATLAB: indMid = 556; % ceil((size(A,1))/2);
        indMid = 556  # hardcoded value
        
        # MATLAB: C = zeros(1,size(A,1));
        # MATLAB: C(1,indMid) = 1;
        C = np.zeros((1, A.shape[0]))
        C[0, indMid - 1] = 1  # MATLAB uses 1-based indexing
        
        # construct linear system object
        # MATLAB: sys = linearSys('heat',A,B,[],C);
        sys = LinearSys('heat', A, B, None, C)
        
        # Parameters --------------------------------------------------------------
        
        # MATLAB: x0 = aux_getInit(A,samples,1);
        x0 = aux_getInit(A, samples, 1)
        # MATLAB: temp = diag(0.1*x0);
        # MATLAB: temp = temp(:,x0 > 0);
        temp = np.diag(0.1 * x0.flatten())
        temp = temp[:, x0.flatten() > 0]
        # MATLAB: params.R0 = zonotope(x0,temp);
        params = {
            'R0': Zonotope(x0, temp)
        }
        # MATLAB: params.U = zonotope(interval(zeros(size(sys.B,2),1)));
        I_U = Interval(np.zeros((sys.B.shape[1], 1)), np.zeros((sys.B.shape[1], 1)))
        params['U'] = I_U.zonotope()
        
        # MATLAB: params.tFinal = 40;
        params['tFinal'] = 40
        
        # MATLAB: options = struct();
        # MATLAB: options.verifyAlg = 'reachavoid:supportFunc';
        options = {
            'verifyAlg': 'reachavoid:supportFunc'
        }
        
        # Specification -----------------------------------------------------------
        
        # forall t: y1 <= 0.02976
        # MATLAB: d = 0.02966+1e-4;
        d = 0.02966 + 1e-4
        # MATLAB: spec = specification(polytope(1,d),'safeSet');
        # according to ARCH report, maximum 0.02966 at time 22.62
        P = Polytope(np.array([[1]]), np.array([[d]]))
        spec = Specification(P, 'safeSet')
        
        # Verification ------------------------------------------------------------
        
        # min steps needed: 2270
        # MATLAB: [res,fals,savedata] = verify(sys,params,options,spec);
        res, fals, savedata = sys.verify(params, options, spec)
        
        # MATLAB: disp("specifications verified: " + res);
        # MATLAB: disp("computation time: " + savedata.tComp);
        print(f"specifications verified: {res}")
        print(f"computation time: {savedata.get('tComp', 'N/A')}")
        
        # Return values -----------------------------------------------------------
        
        # MATLAB: text = ['Heat3D,HEAT02,',num2str(res),',',num2str(savedata.tComp)];
        text = f'Heat3D,HEAT02,{res},{savedata.get("tComp", "N/A")}'
        
        return text


def test_verifyFast_heat3D_HEAT02():
    """Test function for heat3D HEAT02 benchmark verification.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestVerifyFastHeat3DHEAT02()
    result = test.test_verifyFast_heat3D_HEAT02()
    
    print(f"test_verifyFast_heat3D_HEAT02: result = {result}")
    return result


if __name__ == "__main__":
    test_verifyFast_heat3D_HEAT02()

