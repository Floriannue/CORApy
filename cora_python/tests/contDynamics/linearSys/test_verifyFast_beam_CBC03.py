"""
test_verifyFast_beam_CBC03 - beam benchmark from the
    2023 ARCH competition

Syntax:
    text = test_verifyFast_beam_CBC03()

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
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.cartProd_ import cartProd_
from cora_python.contSet.polytope import Polytope
from cora_python.specification.specification import Specification
# verify is a method of linearSys, accessed via sys.verify()


class TestVerifyFastBeamCBC03:
    """Test class for beam CBC03 benchmark verification"""
    
    def test_verifyFast_beam_CBC03(self):
        """Test beam benchmark CBC03 from ARCH 2023 competition"""
        # Model Derivation --------------------------------------------------------
        
        # nodes in model
        # MATLAB: N = 1000;
        N = 1000
        # node of interest
        # MATLAB: node = round(0.7*N);
        node = round(0.7 * N)
        
        # constants
        # MATLAB: rho = 7.3e-4;   % density
        # MATLAB: L = 200;        % length of beam
        # MATLAB: Q = 1;          % cross-section area (renamed from A)
        # MATLAB: E = 30e6;       % Young's modulus
        rho = 7.3e-4  # density
        L = 200  # length of beam
        Q = 1  # cross-section area (renamed from A)
        E = 30e6  # Young's modulus
        
        # MATLAB: ell = L/N;      % length of individual discrete element
        ell = L / N  # length of individual discrete element
        
        # mass matrix (NxN)
        # MATLAB: M = (rho*Q*ell) / 2 * diag([2*ones(N-1,1);1]);
        # MATLAB: Minv = M^(-1);
        M = (rho * Q * ell) / 2 * np.diag(np.concatenate([2 * np.ones(N - 1), [1]]))
        Minv = np.linalg.inv(M)
        
        # load
        # MATLAB: F = zonotope(10000,100);
        F = Zonotope(np.array([[10000]]), np.array([[100]]))
        
        # tridiagonal matrix (NxN)
        # MATLAB: mat = zeros(N);
        # MATLAB: mat(1,1) = 2; mat(1,2) = -1;
        # MATLAB: mat(N,N-1) = -1; mat(N,N) = 1;
        # MATLAB: for r=2:N-1
        # MATLAB:     mat(r,1+(r-2)) = -1;
        # MATLAB:     mat(r,2+(r-2)) = 2;
        # MATLAB:     mat(r,3+(r-2)) = -1;
        # MATLAB: end
        mat = np.zeros((N, N))
        mat[0, 0] = 2
        mat[0, 1] = -1
        mat[N - 1, N - 2] = -1
        mat[N - 1, N - 1] = 1
        for r in range(1, N - 1):
            mat[r, 0 + (r - 1)] = -1
            mat[r, 1 + (r - 1)] = 2
            mat[r, 2 + (r - 1)] = -1
        
        # stiffness matrix (NxN)
        # MATLAB: K = E*Q/ell * mat;
        K = E * Q / ell * mat
        # damping matrix (NxN)
        # MATLAB: a = 1e-6;
        # MATLAB: b = 1e-6;
        # MATLAB: D = a*K + b*M;
        a = 1e-6
        b = 1e-6
        D = a * K + b * M
        
        # state matrix (damped)
        # MATLAB: A = [zeros(N) eye(N); -Minv*K -Minv*D];
        A = np.block([[np.zeros((N, N)), np.eye(N)],
                      [-Minv @ K, -Minv @ D]])
        
        # nr of states
        # MATLAB: dim_x = length(A);
        # Handle sparse matrices - use shape[0] instead of len()
        dim_x = A.shape[0] if hasattr(A, 'shape') else len(A)
        
        # constant inputs
        # MATLAB: A_C = [ A, [zeros(dim_x-1,1);1]; zeros(1,dim_x+1) ];
        # MATLAB: B_C = zeros(size(A_C,1),1);
        A_C = np.block([[A, np.vstack([np.zeros((dim_x - 1, 1)), np.array([[1]])])],
                        [np.zeros((1, dim_x + 1))]])
        B_C = np.zeros((A_C.shape[0], 1))
        
        # Parameters --------------------------------------------------------------
        
        # MATLAB: params.tFinal = 0.01;
        params = {
            'tFinal': 0.01
        }
        
        # initial set: bar at rest
        # MATLAB: params.R0 = cartProd( zonotope(zeros(dim_x,1)), Minv(end,end)*F );
        Z1 = Zonotope(np.zeros((dim_x, 1)), np.zeros((dim_x, 0)))
        Z2 = Zonotope(Minv[-1, -1] * F.c, Minv[-1, -1] * F.G)
        params['R0'] = cartProd_(Z1, Z2)
        
        # input set
        # MATLAB: params.U = zonotope(0);
        params['U'] = Zonotope(np.array([[0]]), np.zeros((1, 0)))
        
        # MATLAB: options = struct();
        # MATLAB: options.verifyAlg = 'reachavoid:supportFunc';
        options = {
            'verifyAlg': 'reachavoid:supportFunc'
        }
        
        # System Dynamics ---------------------------------------------------------
        
        # MATLAB: C = zeros(1,2*N+1);
        # MATLAB: C(1,2*node) = 1;
        C = np.zeros((1, 2 * N + 1))
        C[0, 2 * node] = 1
        
        # construct linear system objects
        # MATLAB: sys = linearSys('beam',A_C,B_C,[],C);
        sys = LinearSys('beam', A_C, B_C, None, C)
        
        # Specification -----------------------------------------------------------
        
        # forall t: y1 <= 74
        # MATLAB: spec = specification(polytope(1,74),'safeSet');
        P = Polytope(np.array([[1]]), np.array([[74]]))
        spec = Specification(P, 'safeSet')
        
        # Verification ------------------------------------------------------------
        
        # min steps needed: 5350
        # MATLAB: [res,fals,savedata] = verify(sys,params,options,spec);
        res, fals, savedata = sys.verify(params, options, spec)
        
        # MATLAB: disp("specifications verified: " + res);
        # MATLAB: disp("computation time: " + savedata.tComp);
        print(f"specifications verified: {res}")
        print(f"computation time: {savedata.get('tComp', 'N/A')}")
        
        # Return value ------------------------------------------------------------
        
        # MATLAB: text = ['Beam,CBC03,',num2str(res),',',num2str(savedata.tComp)];
        text = f'Beam,CBC03,{res},{savedata.get("tComp", "N/A")}'
        
        return text


def test_verifyFast_beam_CBC03():
    """Test function for beam CBC03 benchmark verification.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestVerifyFastBeamCBC03()
    result = test.test_verifyFast_beam_CBC03()
    
    print(f"test_verifyFast_beam_CBC03: result = {result}")
    return result


if __name__ == "__main__":
    test_verifyFast_beam_CBC03()

