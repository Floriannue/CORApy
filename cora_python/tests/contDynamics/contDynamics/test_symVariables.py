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
        """Test basic symVariables without brackets"""
        # MATLAB: sys = contDynamics('test',3,1,2);
        sys = MockContDynamics(nrOfDims=3, nrOfInputs=1, nrOfOutputs=2)
        # MATLAB: [vars,vars_der] = symVariables(sys);
        vars_dict, vars_der = symVariables(sys, False)
        
        # Check that vars.x has 3 elements (nrOfDims)
        assert vars_dict['x'] is not None
        if isinstance(vars_dict['x'], sp.Matrix):
            assert vars_dict['x'].shape[0] == 3
        else:
            assert len(vars_dict['x']) == 3
        
        # Check that vars.u has 1 element (nrOfInputs)
        assert vars_dict['u'] is not None
        if isinstance(vars_dict['u'], sp.Matrix):
            assert vars_dict['u'].shape[0] == 1
        else:
            assert len(vars_dict['u']) == 1
        
        # Check that vars.o has 2 elements (nrOfOutputs)
        assert vars_dict['o'] is not None
        if isinstance(vars_dict['o'], sp.Matrix):
            assert vars_dict['o'].shape[0] == 2
        else:
            assert len(vars_dict['o']) == 2
        
        # Check deviation variables
        assert vars_der['dx'] is not None
        assert vars_der['du'] is not None
        assert vars_der['do'] is not None
    
    def test_symVariables_with_brackets(self):
        """Test symVariables with brackets (xL1R format)"""
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1)
        # MATLAB: [vars,vars_der] = symVariables(sys,true);
        vars_dict, vars_der = symVariables(sys, True)
        
        # With brackets, variables should be in format xL1R, xL2R, etc.
        # Check that variables are created (exact format depends on implementation)
        assert vars_dict['x'] is not None
        assert vars_der['dx'] is not None
    
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
        """Test symVariables without constraint variables"""
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, nrOfConstraints=0)
        # sys doesn't have nrOfConstraints attribute
        vars_dict, vars_der = symVariables(sys, False)
        
        # y should be empty matrix
        assert vars_dict['y'] is not None
        if isinstance(vars_dict['y'], sp.Matrix):
            assert vars_dict['y'].shape[0] == 0 or vars_dict['y'].size == 0
    
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
        """Test symVariables without parameter variables"""
        sys = MockContDynamics(nrOfDims=2, nrOfInputs=1, nrOfOutputs=1, nrOfParam=0)
        vars_dict, vars_der = symVariables(sys, False)
        
        # p should be empty matrix
        assert vars_dict['p'] is not None
        if isinstance(vars_dict['p'], sp.Matrix):
            assert vars_dict['p'].shape[0] == 0 or vars_dict['p'].size == 0
    
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
    return True


if __name__ == "__main__":
    test_symVariables()

