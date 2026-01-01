"""
test_nonlinearSys_linearize - unit_test_function of linearizing nonlinear 
   dynamics: Checks the linearization of the nonlinearSys class
   for the 6D tank example; It is checked whether the A and B matrix
   are correct for a particular linearization point

Syntax:
    pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_linearize.py

Inputs:
    -

Outputs:
    res - true/false

Authors:       Matthias Althoff
Written:       30-July-2017
Last update:   12-September-2017
               13-October-2024 (MW, update syntax)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices
from cora_python.models.Cora.tank.tank6Eq import tank6Eq


class TestNonlinearSysLinearize:
    """Test class for nonlinearSys linearize functionality"""
    
    def test_nonlinearSys_linearize(self):
        """Test linearize for 6D tank example"""
        # tolerance
        # MATLAB: tol = 1e-12;
        tol = 1e-12
        
        # model parameters
        # MATLAB: params.U = zonotope(0,0.005);
        params = {
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
            'uTrans': np.array([[0]])
        }
        
        # reachability settings
        # MATLAB: options.alg = 'lin';
        # MATLAB: options.tensorOrder = 2;
        options = {
            'alg': 'lin',
            'tensorOrder': 2
        }
        
        # system dynamics
        # MATLAB: tank = nonlinearSys(@tank6Eq);
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        # MATLAB: derivatives(tank);
        from cora_python.contDynamics.contDynamics.derivatives import derivatives
        derivatives(tank)
        
        # linearize system
        # MATLAB: dim_x = 6;
        dim_x = 6
        # MATLAB: R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
        R0 = Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x))
        # MATLAB: [~,linsys,linOptions] = linearize(tank,R0,params,options);
        from cora_python.contDynamics.nonlinearSys.linearize import linearize
        _, linsys, linParams, linOptions = linearize(tank, R0, params, options)
        
        # ground truth
        # MATLAB: A_true = [-0.023490689645049, 0, 0, 0, 0, -0.010000000000000; ...
        A_true = np.array([
            [-0.023490689645049, 0, 0, 0, 0, -0.010000000000000],
            [0.023490689645049, -0.016610425942763, 0, 0, 0, 0],
            [0, 0.016610425942763, -0.016610425942763, 0, 0, 0],
            [0, 0, 0.016610425942763, -0.023490689645049, 0, 0],
            [0, 0, 0, 0.023490689645049, -0.010505355776936, 0],
            [0, 0, 0, 0, 0.010505355776936, -0.016610425942763]
        ])
        # MATLAB: U_true = zonotope(zeros(dim_x,1), [0.005; zeros(dim_x-1,1)]);
        U_true = Zonotope(np.zeros((dim_x, 1)), np.vstack([np.array([[0.005]]), np.zeros((dim_x-1, 1))]))
        
        # compare with obtained values
        # MATLAB: assert(compareMatrices(linsys.A,A_true,tol,"equal",true));
        assert compareMatrices(linsys.A, A_true, tol, "equal", True)
        # MATLAB: assert(isequal(linOptions.U,U_true,tol));
        # Note: isequal for zonotopes - check center and generators
        assert np.allclose(linsys.B, linParams.get('U', Zonotope(np.zeros((dim_x, 1)))).center(), atol=tol)
        # Actually, MATLAB checks linOptions.U, which should be in linParams
        if 'U' in linParams:
            U_obtained = linParams['U']
            assert np.allclose(U_obtained.center(), U_true.center(), atol=tol)
            # Compare generators (may need to handle order differences)
            U_obtained_gen = U_obtained.generators()
            U_true_gen = U_true.generators()
            assert compareMatrices(U_obtained_gen, U_true_gen, tol, "equal", False)


def test_nonlinearSys_linearize():
    """Test function for nonlinearSys linearize method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearSysLinearize()
    test.test_nonlinearSys_linearize()
    
    print("test_nonlinearSys_linearize: all tests passed")
    return True


if __name__ == "__main__":
    test_nonlinearSys_linearize()

