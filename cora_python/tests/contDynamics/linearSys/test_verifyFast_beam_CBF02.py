"""
test_verifyFast_beam_CBF02 - beam benchmark from the
    2023 ARCH competition

Syntax:
    text = test_verifyFast_beam_CBF02()

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


class TestVerifyFastBeamCBF02:
    """Test class for beam CBF02 benchmark verification"""
    
    def test_verifyFast_beam_CBF02(self):
        """Test beam benchmark CBF02 from ARCH 2023 competition"""
        # Model Derivation --------------------------------------------------------
        
        # nodes in model
        # MATLAB: N = 500;
        N = 500
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
        
        # mass matrix (diagonal)
        # MATLAB: M = (rho*Q*ell) / 2 * diag([2*ones(N-1,1);1]);
        # MATLAB: Minv = M^(-1);
        M_diag = (rho * Q * ell) / 2 * np.concatenate([2 * np.ones(N - 1), [1]])
        Minv_diag = 1.0 / M_diag
        
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
        # build tridiagonal efficiently (same as MATLAB loop)
        mat = np.diag(2 * np.ones(N))
        mat[-1, -1] = 1
        mat += np.diag(-1 * np.ones(N - 1), -1)
        mat += np.diag(-1 * np.ones(N - 1), 1)
        
        # stiffness matrix (NxN)
        # MATLAB: K = E*Q/ell * mat;
        K = E * Q / ell * mat
        # damping matrix (NxN)
        # MATLAB: a = 1e-6;
        # MATLAB: b = 1e-6;
        # MATLAB: D = a*K + b*M;
        a = 1e-6
        b = 1e-6
        D = a * K + b * np.diag(M_diag)
        
        # state matrix (damped)
        # MATLAB: A = [zeros(N) eye(N); -Minv*K -Minv*D];
        MinvK = Minv_diag[:, None] * K
        MinvD = Minv_diag[:, None] * D
        A = np.block([[np.zeros((N, N)), np.eye(N)],
                      [-MinvK, -MinvD]])
        
        # Parameters --------------------------------------------------------------
        
        # MATLAB: params.tFinal = 0.01;
        params = {
            'tFinal': 0.01
        }
        
        # nr of states
        # MATLAB: dim_x = length(A);
        # Handle sparse matrices - use shape[0] instead of len()
        dim_x = A.shape[0] if hasattr(A, 'shape') else len(A)
        
        # initial set: bar at rest
        # MATLAB: params.R0 = zonotope(zeros(dim_x,1));
        params['R0'] = Zonotope(np.zeros((dim_x, 1)), np.zeros((dim_x, 0)))
        
        # input set
        # MATLAB: params.U = cartProd( zonotope(zeros(dim_x-1,1)), Minv(end,end)*F );
        Z1 = Zonotope(np.zeros((dim_x - 1, 1)), np.zeros((dim_x - 1, 0)))
        Z2 = Zonotope(Minv_diag[-1] * F.c, Minv_diag[-1] * F.G)
        params['U'] = cartProd_(Z1, Z2)
        
        # MATLAB: options = struct();
        # MATLAB: options.verifyAlg = 'reachavoid:supportFunc';
        options = {
            'verifyAlg': 'reachavoid:supportFunc'
        }
        
        # System Dynamics ---------------------------------------------------------
        
        # MATLAB: C = zeros(1,2*N);
        # MATLAB: C(1,2*node) = 1;
        C = np.zeros((1, 2 * N))
        C[0, 2 * node] = 1
        
        # construct linear system objects
        # MATLAB: sys = linearSys('beam',A,1,[],C);
        # MATLAB: scalar B means inputs = states (B gets converted to B*eye(states) internally)
        # So we should pass scalar 1, not a matrix
        sys = LinearSys('beam', A, 1, None, C)
        
        # Specification -----------------------------------------------------------
        
        # forall t: y1 <= 74
        # MATLAB: spec = specification(polytope(1,74),'safeSet');
        P = Polytope(np.array([[1]]), np.array([[74]]))
        spec = Specification(P, 'safeSet')
        
        # Verification ------------------------------------------------------------
        
        # min steps needed: 2070
        # MATLAB: [res,fals,savedata] = verify(sys,params,options,spec);
        res, fals, savedata = sys.verify(params, options, spec)
        
        # MATLAB: disp("specifications verified: " + res);
        # MATLAB: disp("computation time: " + savedata.tComp);
        print(f"specifications verified: {res}")
        print(f"computation time: {savedata.get('tComp', 'N/A')}")
        
        # Verify the result
        # MATLAB expects: res = True (safeSet satisfied)
        assert res is not None, "Verification result should not be None"
        assert res == True, f"Specification should be satisfied (safeSet), got {res}"
        assert savedata.get('tComp', 0) > 0, "Computation time should be positive"


def test_verifyFast_beam_CBF02():
    """Test function for beam CBF02 benchmark verification.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestVerifyFastBeamCBF02()
    test.test_verifyFast_beam_CBF02()


if __name__ == "__main__":
    test_verifyFast_beam_CBF02()

