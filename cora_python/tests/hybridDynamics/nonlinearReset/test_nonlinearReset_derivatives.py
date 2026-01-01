"""
test_nonlinearReset_derivatives - test function for derivative
   computation of nonlinear reset functions

Syntax:
    res = test_nonlinearReset_derivatives

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       12-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import os
from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
from cora_python.hybridDynamics.nonlinearReset.derivatives import derivatives


class TestNonlinearResetDerivatives:
    """Test class for nonlinearReset derivatives functionality"""
    
    def test_nonlinearReset_derivatives(self):
        """Test derivative computation for nonlinear reset functions"""
        # set path to this folder
        # MATLAB: path = mfilename('fullpath');
        # MATLAB: idxFilesep = strfind(path,filesep);
        # MATLAB: path = path(1:idxFilesep(end));
        path = os.path.dirname(os.path.abspath(__file__))
        fname = 'test_nonlinearReset_derivatives_generatedfile'
        # fullname needs to match generated file, otherwise delete won't work
        # MATLAB: fullname_jacobian = [path filesep fname '_jacobian.m'];
        # MATLAB: fullname_hessian = [path filesep fname '_hessian.m'];
        # MATLAB: fullname_thirdorder = [path filesep fname 'thirdOrderTensor_.m'];
        fullname_jacobian = os.path.join(path, f'{fname}_jacobian.py')
        fullname_hessian = os.path.join(path, f'{fname}_hessian.py')
        fullname_thirdorder = os.path.join(path, f'{fname}thirdOrderTensor_.py')
        
        # empty
        # MATLAB: nonlinReset = nonlinearReset();
        # MATLAB: assert(nonlinReset.preStateDim == 0);
        # MATLAB: assert(nonlinReset.inputDim == 1);
        # MATLAB: assert(nonlinReset.postStateDim == 0);
        # Note: Empty case is commented out in MATLAB test
        
        # only states
        # MATLAB: f = @(x,u) [x(1)*x(2); x(2)];
        def f1(x, u):
            return np.array([[x[0, 0] * x[1, 0]], [x[1, 0]]])
        
        # only Jacobian and Hessian
        # MATLAB: nonlinReset = nonlinearReset(f);
        nonlinReset = NonlinearReset(f1)
        # MATLAB: nonlinReset = derivatives(nonlinReset,path,fname,2);
        nonlinReset = derivatives(nonlinReset, path, fname, 2)
        # MATLAB: assert(~isempty(nonlinReset.J));
        # MATLAB: assert(~isempty(nonlinReset.H));
        assert nonlinReset.J is not None
        assert nonlinReset.H is not None
        # MATLAB: delete(fullname_jacobian);
        # MATLAB: delete(fullname_hessian);
        if os.path.exists(fullname_jacobian):
            os.remove(fullname_jacobian)
        if os.path.exists(fullname_hessian):
            os.remove(fullname_hessian)
        
        # states and inputs
        # MATLAB: f = @(x,u) [x(1) - u(1); x(1)*x(2)];
        def f2(x, u):
            return np.array([[x[0, 0] - u[0, 0]], [x[0, 0] * x[1, 0]]])
        
        # MATLAB: nonlinReset = nonlinearReset(f);
        nonlinReset = NonlinearReset(f2)
        # MATLAB: nonlinReset = derivatives(nonlinReset,path,fname,2);
        nonlinReset = derivatives(nonlinReset, path, fname, 2)
        # MATLAB: assert(~isempty(nonlinReset.J));
        # MATLAB: assert(~isempty(nonlinReset.H));
        assert nonlinReset.J is not None
        assert nonlinReset.H is not None
        # MATLAB: delete(fullname_jacobian);
        # MATLAB: delete(fullname_hessian);
        if os.path.exists(fullname_jacobian):
            os.remove(fullname_jacobian)
        if os.path.exists(fullname_hessian):
            os.remove(fullname_hessian)
        
        # states and inputs, different output dimension
        # MATLAB: f = @(x,u) x(1)*x(2) - u(1);
        def f3(x, u):
            return np.array([[x[0, 0] * x[1, 0] - u[0, 0]]])
        
        # MATLAB: nonlinReset = nonlinearReset(f);
        nonlinReset = NonlinearReset(f3)
        # MATLAB: nonlinReset = derivatives(nonlinReset,path,fname,2);
        nonlinReset = derivatives(nonlinReset, path, fname, 2)
        # MATLAB: assert(~isempty(nonlinReset.J));
        # MATLAB: assert(~isempty(nonlinReset.H));
        assert nonlinReset.J is not None
        assert nonlinReset.H is not None
        # MATLAB: delete(fullname_jacobian);
        # MATLAB: delete(fullname_hessian);
        if os.path.exists(fullname_jacobian):
            os.remove(fullname_jacobian)
        if os.path.exists(fullname_hessian):
            os.remove(fullname_hessian)
        
        # test completed
        # MATLAB: res = true;
        res = True
        
        return res


def test_nonlinearReset_derivatives():
    """Test function for nonlinearReset derivatives method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestNonlinearResetDerivatives()
    result = test.test_nonlinearReset_derivatives()
    
    print("test_nonlinearReset_derivatives: all tests passed")
    return result


if __name__ == "__main__":
    test_nonlinearReset_derivatives()

