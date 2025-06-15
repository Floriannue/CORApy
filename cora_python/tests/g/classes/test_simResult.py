"""
test_simResult - unit test function for simResult class

Tests the simResult class for storing simulation results.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.classes.simResult import SimResult


class TestSimResult:
    def test_simResult_constructor_empty(self):
        """Test empty constructor"""
        simRes = SimResult()
        assert simRes.x == []
        assert simRes.t == []
        assert simRes.y == []
        assert simRes.a == []
        assert simRes.loc == 0
    
    def test_simResult_constructor_basic(self):
        """Test constructor with basic inputs"""
        x = [np.array([[1, 2], [3, 4]])]  # One trajectory, 2 time steps, 2 states
        t = [np.array([[0], [0.1]])]      # Corresponding time points
        
        simRes = SimResult(x, t)
        assert simRes.x == x
        assert simRes.t == t
        assert simRes.y == []
        assert simRes.a == []
        assert simRes.loc == 0
    
    def test_simResult_constructor_full(self):
        """Test constructor with all inputs"""
        x = [np.array([[1, 2], [3, 4]])]  # States
        t = [np.array([[0], [0.1]])]      # Time
        y = [np.array([[0.5], [1.5]])]    # Outputs
        a = [np.array([[0.1], [0.2]])]    # Algebraic variables
        loc = 1
        
        simRes = SimResult(x, t, loc, y, a)
        assert simRes.x == x
        assert simRes.t == t
        assert simRes.y == y
        assert simRes.a == a
        assert simRes.loc == loc
    
    def test_simResult_validation_x_t_length(self):
        """Test validation of x and t having same length"""
        x = [np.array([[1, 2]]), np.array([[3, 4]])]  # 2 trajectories
        t = [np.array([[0]])]                          # 1 trajectory
        
        with pytest.raises(ValueError, match="x and t must have the same length"):
            SimResult(x, t)
    
    def test_simResult_validation_y_length(self):
        """Test validation of y having same length as x"""
        x = [np.array([[1, 2]])]
        t = [np.array([[0]])]
        y = [np.array([[0.5]]), np.array([[1.5]])]  # Wrong length
        
        with pytest.raises(ValueError, match="y must have the same length as x"):
            SimResult(x, t, 0, y)
    
    def test_simResult_validation_a_length(self):
        """Test validation of a having same length as x"""
        x = [np.array([[1, 2]])]
        t = [np.array([[0]])]
        a = [np.array([[0.1]]), np.array([[0.2]])]  # Wrong length
        
        with pytest.raises(ValueError, match="a must have the same length as x"):
            SimResult(x, t, 0, [], a)
    
    def test_simResult_validation_trajectory_time_steps(self):
        """Test validation of trajectory time steps"""
        x = [np.array([[1, 2], [3, 4]])]  # 2 time steps
        t = [np.array([[0]])]             # 1 time step
        
        with pytest.raises(ValueError, match="x and t must have same number of time steps"):
            SimResult(x, t)
    
    def test_len(self):
        """Test __len__ method"""
        # Empty
        simRes = SimResult()
        assert len(simRes) == 0
        
        # With trajectories
        x = [np.array([[1, 2]]), np.array([[3, 4]])]
        t = [np.array([[0]]), np.array([[0.1]])]
        simRes = SimResult(x, t)
        assert len(simRes) == 2
    
    def test_is_empty(self):
        """Test is_empty method"""
        # Empty
        simRes = SimResult()
        assert simRes.is_empty()
        
        # Non-empty
        x = [np.array([[1, 2]])]
        t = [np.array([[0]])]
        simRes = SimResult(x, t)
        assert not simRes.is_empty()
    
    def test_str_representation(self):
        """Test string representation"""
        # Empty
        simRes = SimResult()
        str_repr = str(simRes)
        assert "SimResult: empty" in str_repr
        
        # With data
        x = [np.array([[1, 2, 3]])]  # 1 trajectory, 1 time step, 3 states
        t = [np.array([[0]])]
        y = [np.array([[0.5, 1.5]])]  # 2 outputs
        a = [np.array([[0.1]])]       # 1 algebraic variable
        simRes = SimResult(x, t, 0, y, a)
        
        str_repr = str(simRes)
        assert "SimResult: 1 trajectories" in str_repr
        assert "3 states" in str_repr
        assert "2 outputs" in str_repr
        assert "1 algebraic variables" in str_repr
    
    def test_addition(self):
        """Test addition operator"""
        # First simResult
        x1 = [np.array([[1, 2]])]
        t1 = [np.array([[0]])]
        y1 = [np.array([[0.5]])]
        simRes1 = SimResult(x1, t1, 1, y1)
        
        # Second simResult
        x2 = [np.array([[3, 4]])]
        t2 = [np.array([[0.1]])]
        y2 = [np.array([[1.5]])]
        simRes2 = SimResult(x2, t2, 2, y2)
        
        # Add them
        simRes_sum = simRes1 + simRes2
        
        assert len(simRes_sum.x) == 2
        assert len(simRes_sum.t) == 2
        assert len(simRes_sum.y) == 2
        assert simRes_sum.loc == [1, 2]
        
        # Check data integrity
        assert np.allclose(simRes_sum.x[0], x1[0])
        assert np.allclose(simRes_sum.x[1], x2[0])
    
    def test_addition_type_error(self):
        """Test addition with wrong type"""
        simRes = SimResult()
        
        with pytest.raises(TypeError, match="Can only add SimResult objects"):
            simRes + "invalid"
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication"""
        x = [np.array([[1, 2], [3, 4]])]
        t = [np.array([[0], [0.1]])]
        y = [np.array([[0.5], [1.5]])]
        simRes = SimResult(x, t, 0, y)
        
        # Multiply by 2
        simRes_mult = simRes * 2
        
        expected_x = [np.array([[2, 4], [6, 8]])]
        expected_y = [np.array([[1.0], [3.0]])]
        
        assert np.allclose(simRes_mult.x[0], expected_x[0])
        assert np.allclose(simRes_mult.y[0], expected_y[0])
        assert np.allclose(simRes_mult.t[0], t[0])  # Time unchanged
    
    def test_right_scalar_multiplication(self):
        """Test right scalar multiplication"""
        x = [np.array([[1, 2]])]
        t = [np.array([[0]])]
        simRes = SimResult(x, t)
        
        # Right multiplication
        simRes_mult = 3 * simRes
        
        expected_x = [np.array([[3, 6]])]
        assert np.allclose(simRes_mult.x[0], expected_x[0])
    
    def test_scalar_multiplication_type_error(self):
        """Test scalar multiplication with wrong type"""
        simRes = SimResult()
        
        with pytest.raises(TypeError, match="Can only multiply by scalar"):
            simRes * "invalid"
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        x = [np.array([[1, 2, 3], [4, 5, 6]])]  # 2 time steps, 3 states
        t = [np.array([[0], [0.1]])]
        y = [np.array([[0.5, 1.5], [2.5, 3.5]])]  # 2 outputs (different dimension)
        simRes = SimResult(x, t, 0, y)
        
        # Multiply by 2x3 matrix to get 2D output
        matrix = np.array([[1, 0, 1], [0, 1, 0]])
        simRes_mult = simRes @ matrix
        
        # Expected: [1+3, 2] = [4, 2] and [4+6, 5] = [10, 5]
        expected_x = [np.array([[4, 2], [10, 5]])]
        
        assert np.allclose(simRes_mult.x[0], expected_x[0])
        assert np.allclose(simRes_mult.t[0], t[0])  # Time unchanged
        
        # Outputs should be unchanged since dimensions don't match (2 outputs vs 3 states)
        assert np.allclose(simRes_mult.y[0], y[0])
        
        # Algebraic variables should be copied unchanged
        assert simRes_mult.a == simRes.a
    
    def test_multiple_trajectories(self):
        """Test with multiple trajectories"""
        x = [
            np.array([[1, 2], [3, 4]]),  # Trajectory 1
            np.array([[5, 6], [7, 8]])   # Trajectory 2
        ]
        t = [
            np.array([[0], [0.1]]),      # Time for trajectory 1
            np.array([[0], [0.1]])       # Time for trajectory 2
        ]
        
        simRes = SimResult(x, t)
        
        assert len(simRes) == 2
        assert not simRes.is_empty()
        assert np.allclose(simRes.x[0], x[0])
        assert np.allclose(simRes.x[1], x[1])


if __name__ == "__main__":
    pytest.main([__file__]) 