"""
test_interval_minus - unit test function for interval minus operation

This module tests the minus operation for intervals, following the MATLAB
test patterns from test_interval_minus.m.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import cora_python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import Interval
from cora_python.g.functions.matlab.validate.postprocessing import CORAError


class TestIntervalMinus:
    """Test class for interval minus operation"""
    
    def test_minus_scalar_equal(self):
        """Test minus operation with equal scalar intervals"""
        tol = 1e-9
        
        a = Interval([0], [0])
        b = Interval([1], [1])
        c = a - b
        
        assert abs(c.inf[0] + 1.0) <= tol
        assert abs(c.sup[0] + 1.0) <= tol
        
        # Reverse operation
        c = b - a
        assert abs(c.inf[0] - 1.0) <= tol
        assert abs(c.sup[0] - 1.0) <= tol
    
    def test_minus_scalar_one_negative(self):
        """Test minus operation with one negative interval"""
        tol = 1e-9
        
        a = Interval([-1], [0])
        b = Interval([-1], [0])
        c = a - b
        
        assert abs(c.inf[0] + 1.0) <= tol
        assert abs(c.sup[0] - 1.0) <= tol
        
        # Different bounds
        a = Interval([-1], [0])
        b = Interval([-2], [0])
        c = a - b
        
        assert abs(c.inf[0] + 1.0) <= tol
        assert abs(c.sup[0] - 2.0) <= tol
        
        # With float values
        a = Interval([-1.0], [0])
        b = Interval([-2.0], [0])
        c = a - b
        
        assert abs(c.inf[0] + 1.0) <= tol
        assert abs(c.sup[0] - 2.0) <= tol
    
    def test_minus_scalar_both_positive(self):
        """Test minus operation with both positive intervals"""
        tol = 1e-9
        
        a = Interval([0], [1])
        b = Interval([0], [1])
        c = a - b
        
        assert abs(c.inf[0] + 1.0) <= tol
        assert abs(c.sup[0] - 1.0) <= tol
        
        # Different bounds
        a = Interval([0], [1])
        b = Interval([0], [2])
        c = a - b
        
        assert abs(c.inf[0] + 2.0) <= tol
        assert abs(c.sup[0] - 1.0) <= tol
        
        # With float values
        a = Interval([0], [1.0])
        b = Interval([0], [2.0])
        c = a - b
        
        assert abs(c.inf[0] + 2.0) <= tol
        assert abs(c.sup[0] - 1.0) <= tol
    
    def test_minus_scalar_mixed(self):
        """Test minus operation with mixed sign intervals"""
        tol = 1e-9
        
        a = Interval([-2.0], [2.0])
        b = Interval([-3.0], [2.0])
        c = a - b
        
        assert abs(c.inf[0] + 4.0) <= tol
        assert abs(c.sup[0] - 5.0) <= tol
        
        # Reverse operation
        c = b - a
        assert abs(c.inf[0] + 5.0) <= tol
        assert abs(c.sup[0] - 4.0) <= tol
    
    def test_minus_scalar_numeric(self):
        """Test minus operation between interval and numeric"""
        tol = 1e-9
        
        # Numeric - interval
        a = Interval([-2.0], [1.0])
        c = 1 - a
        
        assert abs(c.inf[0] + 0.0) <= tol
        assert abs(c.sup[0] - 3.0) <= tol
        
        # Interval - numeric
        c = a - 1
        assert abs(c.inf[0] + 3.0) <= tol
        assert abs(c.sup[0] - 0.0) <= tol
    
    def test_minus_vector(self):
        """Test minus operation with vector intervals"""
        tol = 1e-9
        
        a = Interval([-5.0, -4.0, -3, 0, 0, 5], [-2, 0.0, 2.0, 0, 5, 8])
        b = Interval([-6.1, -4.5, -3.3, 0, 0, 5], [-2.2, 0.0, 2.8, 0, 5.7, 8.2])
        c = a - b
        
        # Check each component
        assert abs(c.inf[0] + 2.8) <= tol and abs(c.sup[0] - 4.1) <= tol
        assert abs(c.inf[1] + 4.0) <= tol and abs(c.sup[1] - 4.5) <= tol
        assert abs(c.inf[2] + 5.8) <= tol and abs(c.sup[2] - 5.3) <= tol
        assert abs(c.inf[3] + 0.0) <= tol and abs(c.sup[3] - 0.0) <= tol
        assert abs(c.inf[4] + 5.7) <= tol and abs(c.sup[4] - 5.0) <= tol
        assert abs(c.inf[5] + 3.2) <= tol and abs(c.sup[5] - 3.0) <= tol
    
    def test_minus_empty_Interval(self):
        """Test minus operation with empty intervals"""
        I = Interval.empty(1)
        v = 1
        
        # Empty - numeric should be empty
        I_minus = I - v
        assert I_minus.representsa_('emptySet')
        
        # Numeric - empty should be empty
        I_minus = v - I
        assert I_minus.representsa_('emptySet')
    
    def test_minus_bounded_intervals(self):
        """Test minus operation between bounded intervals"""
        I1 = Interval([-2, -1], [3, 4])
        I2 = Interval([-1, -3], [1, -1])
        I_minus = I1 - I2
        
        # Expected: [-2,3] - [-1,1] = [-2-1, 3-(-1)] = [-3, 4]
        #           [-1,4] - [-3,-1] = [-1-(-1), 4-(-3)] = [0, 7]
        expected = Interval([-3, 0], [4, 7])
        assert I_minus == expected
    
    def test_minus_unbounded_intervals(self):
        """Test minus operation with unbounded intervals"""
        I1 = Interval([-np.inf, -2], [2, 4])
        I2 = Interval([-1, 0], [1, np.inf])
        I_minus = I1 - I2
        
        # Expected: [-inf,2] - [-1,1] = [-inf-1, 2-(-1)] = [-inf, 3]
        #           [-2,4] - [0,inf] = [-2-inf, 4-0] = [-inf, 4]
        assert np.isinf(I_minus.inf[0]) and I_minus.inf[0] < 0
        assert np.isclose(I_minus.sup[0], 3)
        assert np.isinf(I_minus.inf[1]) and I_minus.inf[1] < 0
        assert np.isclose(I_minus.sup[1], 4)
    
    def test_minus_matrix_intervals(self):
        """Test minus operation with matrix intervals"""
        I1 = Interval([[-1, 0], [1, -1]], [[2, 1], [3, 2]])
        I2 = Interval([[0, -1], [-1, 0]], [[1, 0], [2, 1]])
        
        I_minus = I1 - I2
        # Expected calculation for each element:
        # [-1,2] - [0,1] = [-1-1, 2-0] = [-2, 2]
        # [0,1] - [-1,0] = [0-0, 1-(-1)] = [0, 2]
        # [1,3] - [-1,2] = [1-2, 3-(-1)] = [-1, 4]
        # [-1,2] - [0,1] = [-1-1, 2-0] = [-2, 2]
        expected = Interval([[-2, 0], [-1, -2]], [[2, 2], [4, 2]])
        assert I_minus == expected
    
    def test_minus_vector_numeric(self):
        """Test minus operation between vector interval and numeric"""
        I = Interval([-1, 0], [2, 3])
        
        # Interval - scalar
        I_minus = I - 1
        expected = Interval([-2, -1], [1, 2])
        assert I_minus == expected
        
        # Scalar - interval
        I_minus = 5 - I
        expected = Interval([3, 2], [6, 5])
        assert I_minus == expected
    
    def test_minus_vector_array(self):
        """Test minus operation between interval and array"""
        I = Interval([-1, 0], [2, 3])
        v = np.array([1, -1])
        
        I_minus = I - v
        expected = Interval([-2, 1], [1, 4])
        assert I_minus == expected
        
        # Reverse operation
        I_minus = v - I
        expected = Interval([-1, -4], [2, -1])
        assert I_minus == expected
    
    def test_minus_zero_operations(self):
        """Test minus operation with zero"""
        I = Interval([-1, 0], [2, 3])
        
        # Subtract zero
        I_minus = I - 0
        assert I_minus == I
        
        # Zero minus interval
        I_minus = 0 - I
        expected = Interval([-2, -3], [1, 0])
        assert I_minus == expected
    
    def test_minus_error_cases(self):
        """Test error cases for minus operation"""
        # Both operands are numeric (should raise error)
        with pytest.raises(CORAError):
            from cora_python.contSet.Interval.minus import minus
            minus(5, 3)


if __name__ == '__main__':
    pytest.main([__file__]) 
