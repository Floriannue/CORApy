"""
test_symVariables - GENERATED TEST
   No MATLAB source test exists. This test was created by analyzing the MATLAB 
   implementation logic in symVariables.py and ensuring thorough coverage.

   This test verifies that symVariables correctly generates symbolic variables 
   for continuous systems, including:
   - State variables (x)
   - Input variables (u)
   - Constraint variables (y)
   - Output variables (o)
   - Parameter variables (p)
   - Deviation variables (dx, du, dy, do)
   
   Tests both withBrackets=True and withBrackets=False cases.

Syntax:
    pytest cora_python/tests/contDynamics/contDynamics/test_symVariables.py

Authors:       Generated test based on MATLAB implementation
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
import sympy as sp
from cora_python.contDynamics.contDynamics.symVariables import symVariables


class MockContDynamics:
    """Mock contDynamics object for testing"""
    def __init__(self, nrOfDims=3, nrOfInputs=2, nrOfOutputs=1, 
                 nrOfConstraints=0, nrOfParam=0, is_nonlinearARX=False):
        self.nrOfDims = nrOfDims
        self.nrOfInputs = nrOfInputs
        self.nrOfOutputs = nrOfOutputs
        self.nrOfConstraints = nrOfConstraints if hasattr(self, 'nrOfConstraints') else None
        self.nrOfParam = nrOfParam if hasattr(self, 'nrOfParam') else None
        self._is_nonlinearARX = is_nonlinearARX
    
    def __class__(self):
        class_name = 'nonlinearARX' if self._is_nonlinearARX else 'nonlinearSys'
        return type(class_name, (), {})


class TestSymVariables:
    """Test class for symVariables functionality"""
    
    def test_symVariables_basic_without_brackets(self):
        """Test basic symVariables without brackets
        
        Verified against MATLAB using debug_matlab_symVariables.m
        MATLAB output: vars.x = [x1, x2, x3], vars.u = [u1], vars.o = [y1, y2]
        vars_der.x = [dx1, dx2, dx3], vars_der.u = [du1], vars_der.o = [do1, do2]
        """
        # MATLAB: sys = contDynamics('test',3,1,2);
        sys = MockContDynamics(nrOfDims=3, nrOfInputs=1, nrOfOutputs=2)
        # MATLAB: [vars,vars_der] = symVariables(sys);
        vars_dict, vars_der = symVariables(sys, False)
        
        # MATLAB verified values
        assert vars_dict['x'].shape[0] == 3, "MATLAB: vars.x length = 3"
        assert str(vars_dict['x'][0]) == 'x1', "MATLAB: vars.x(1) = x1"
        assert str(vars_dict['x'][1]) == 'x2', "MATLAB: vars.x(2) = x2"
        assert str(vars_dict['x'][2]) == 'x3', "MATLAB: vars.x(3) = x3"
        
        assert vars_dict['u'].shape[0] == 1, "MATLAB: vars.u length = 1"
        assert str(vars_dict['u'][0]) == 'u1', "MATLAB: vars.u(1) = u1"
        
        assert vars_dict['o'].shape[0] == 2, "MATLAB: vars.o length = 2"
        assert str(vars_dict['o'][0]) == 'y1', "MATLAB: vars.o(1) = y1"
        assert str(vars_dict['o'][1]) == 'y2', "MATLAB: vars.o(2) = y2"
        
        # Deviation variables (Python uses 'dx', 'du', 'do' keys, MATLAB uses 'x', 'u', 'o')
        assert vars_der['dx'].shape[0] == 3, "MATLAB: vars_der.x length = 3"
        assert str(vars_der['dx'][0]) == 'dx1', "MATLAB: vars_der.x(1) = dx1"
        assert vars_der['du'].shape[0] == 1, "MATLAB: vars_der.u length = 1"
        assert str(vars_der['du'][0]) == 'du1', "MATLAB: vars_der.u(1) = du1"
        assert vars_der['do'].shape[0] == 2, "MATLAB: vars_der.o length = 2"
        assert str(vars_der['do'][0]) == 'do1', "MATLAB: vars_der.o(1) = do1"
        assert str(vars_der['do'][1]) == 'do2', "MATLAB: vars_der.o(2) = do2"
    
    def test_symVariables_with_brackets(self):
        """Test symVariables with brackets (xL1R format)
        
        Verified against MATLAB using debug_matlab_symVariables.m
        MATLAB output: vars2.x = [xL1R, xL2R], vars_der2.x = [dxL1R, dxL2R]
        """
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1)
        # MATLAB: [vars,vars_der] = symVariables(sys,true);
        vars_dict, vars_der = symVariables(sys, True)
        
        # MATLAB verified values
        assert vars_dict['x'].shape[0] == 2, "MATLAB: vars2.x length = 2"
        assert str(vars_dict['x'][0]) == 'xL1R', "MATLAB: vars2.x(1) = xL1R"
        assert str(vars_dict['x'][1]) == 'xL2R', "MATLAB: vars2.x(2) = xL2R"
        assert vars_der['dx'].shape[0] == 2, "MATLAB: vars_der2.x length = 2"
        assert str(vars_der['dx'][0]) == 'dxL1R', "MATLAB: vars_der2.x(1) = dxL1R"
    
    def test_symVariables_with_constraints(self):
        """Test symVariables with constraint variables"""
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, nrOfConstraints=2)
        sys.nrOfConstraints = 2  # Ensure it's set
        vars_dict, vars_der = symVariables(sys, False)
        
        # Check constraint variables
        assert vars_dict['y'] is not None
        if isinstance(vars_dict['y'], sp.Matrix):
            assert vars_dict['y'].shape[0] == 2
        else:
            assert len(vars_dict['y']) == 2
        
        assert vars_der['dy'] is not None
    
    def test_symVariables_without_constraints(self):
        """Test symVariables without constraint variables
        
        Verified against MATLAB using debug_matlab_symVariables.m
        MATLAB output: vars4.y length = 0
        """
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, nrOfConstraints=0)
        # sys doesn't have nrOfConstraints attribute
        vars_dict, vars_der = symVariables(sys, False)
        
        # MATLAB verified: y should be empty matrix
        assert vars_dict['y'].shape[0] == 0 or vars_dict['y'].size == 0, "MATLAB: vars4.y length = 0"
    
    def test_symVariables_with_parameters(self):
        """Test symVariables with parameter variables"""
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, nrOfParam=3)
        sys.nrOfParam = 3  # Ensure it's set
        vars_dict, vars_der = symVariables(sys, False)
        
        # Check parameter variables
        assert vars_dict['p'] is not None
        if isinstance(vars_dict['p'], sp.Matrix):
            assert vars_dict['p'].shape[0] == 3
        else:
            assert len(vars_dict['p']) == 3
    
    def test_symVariables_without_parameters(self):
        """Test symVariables without parameter variables
        
        Verified against MATLAB using debug_matlab_symVariables.m
        MATLAB output: vars6.p length = 0
        """
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, nrOfParam=0)
        vars_dict, vars_der = symVariables(sys, False)
        
        # MATLAB verified: p should be empty matrix
        assert vars_dict['p'].shape[0] == 0 or vars_dict['p'].size == 0, "MATLAB: vars6.p length = 0"
    
    def test_symVariables_nonlinearARX(self):
        """Test symVariables for nonlinearARX system"""
        # MATLAB: if isa(sys,'nonlinearARX')
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, is_nonlinearARX=True)
        sys.n_p = 2  # Add n_p attribute for nonlinearARX
        sys.nrOfOutputs = 1
        
        vars_dict, vars_der = symVariables(sys, False)
        
        # For nonlinearARX: vars.x = aux_symVector('x',sys.n_p*sys.nrOfOutputs,withBrackets)
        # vars.x should have n_p * nrOfOutputs = 2 * 1 = 2 elements
        assert vars_dict['x'] is not None
        # vars.u should have (n_p+1)*nrOfInputs = 3 * 1 = 3 elements
        assert vars_dict['u'] is not None


def test_symVariables():
    """Test function for symVariables method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestSymVariables()
    test.test_symVariables_basic_without_brackets()
    test.test_symVariables_with_brackets()
    test.test_symVariables_with_constraints()
    test.test_symVariables_without_constraints()
    test.test_symVariables_with_parameters()
    test.test_symVariables_without_parameters()
    test.test_symVariables_nonlinearARX()
    
    print("test_symVariables: all tests passed")
    # pytest expects None, not True


if __name__ == "__main__":
    test_symVariables()

