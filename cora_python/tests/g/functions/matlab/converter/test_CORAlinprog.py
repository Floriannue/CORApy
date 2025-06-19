"""
test_CORAlinprog - unit test function for CORAlinprog

Tests the CORAlinprog function for solving linear programs.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.converter import CORAlinprog


class TestCORAlinprog:
    def test_CORAlinprog_basic(self):
        """Test basic linear programming problem"""
        # min x1 + 2*x2
        # s.t. x1 + x2 <= 3
        #      x1, x2 >= 0
        problem = {
            'f': [1, 2],
            'Aineq': [[1, 1]],
            'bineq': [3],
            'lb': [0, 0]
        }
        
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # Should find optimal solution
        assert exitflag == 1
        assert x is not None
        assert fval is not None
        assert x.shape == (2, 1)
        
        # Optimal solution should be x1=0, x2=0 with fval=0 (origin is optimal)
        assert np.isclose(fval, 0.0, atol=1e-6)
        assert np.isclose(x[0, 0], 0.0, atol=1e-6)
        assert np.isclose(x[1, 0], 0.0, atol=1e-6)
    
    def test_CORAlinprog_with_equality(self):
        """Test with equality constraints"""
        # min x1 + x2
        # s.t. x1 + x2 = 2
        #      x1, x2 >= 0
        problem = {
            'f': [1, 1],
            'Aeq': [[1, 1]],
            'beq': [2],
            'lb': [0, 0]
        }
        
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # Should find optimal solution
        assert exitflag == 1
        assert x is not None
        assert fval is not None
        
        # Optimal value should be 2
        assert np.isclose(fval, 2.0, atol=1e-6)
        
        # Solution should satisfy equality constraint
        assert np.isclose(x[0, 0] + x[1, 0], 2.0, atol=1e-6)
    
    def test_CORAlinprog_infeasible(self):
        """Test infeasible problem"""
        # min x1
        # s.t. x1 <= -1
        #      x1 >= 0
        problem = {
            'f': [1],
            'Aineq': [[1]],
            'bineq': [-1],
            'lb': [0]
        }
        
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # Should detect infeasibility
        assert exitflag == -2
        assert x is None
        assert fval is None
    
    def test_CORAlinprog_unbounded(self):
        """Test unbounded problem"""
        # min -x1
        # s.t. x1 >= 0 (no upper bound)
        problem = {
            'f': [-1],
            'lb': [0]
        }
        
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # Should detect unboundedness or find large negative value
        # (behavior may vary by solver)
        assert exitflag in [-3, 1]  # Either unbounded or found large solution
    
    def test_CORAlinprog_bounds(self):
        """Test with upper and lower bounds"""
        # min x1 + x2
        # s.t. 1 <= x1 <= 2
        #      0 <= x2 <= 1
        problem = {
            'f': [1, 1],
            'lb': [1, 0],
            'ub': [2, 1]
        }
        
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # Should find optimal solution at lower bounds
        assert exitflag == 1
        assert x is not None
        assert np.isclose(x[0, 0], 1.0, atol=1e-6)
        assert np.isclose(x[1, 0], 0.0, atol=1e-6)
        assert np.isclose(fval, 1.0, atol=1e-6)
    
    def test_CORAlinprog_empty_problem(self):
        """Test with minimal problem specification"""
        # min x1
        # (no constraints)
        problem = {
            'f': [1]
        }
        
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        # Should handle gracefully (may be unbounded)
        assert exitflag in [-3, 0, 1]  # Various possible outcomes


if __name__ == "__main__":
    pytest.main([__file__]) 