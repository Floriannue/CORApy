"""
test_interval_plus - unit test function for interval plus operation

This module tests the plus operation (Minkowski sum) for intervals,
including addition with other intervals and numeric values.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import cora_python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestIntervalPlus:
    """Test class for interval plus operation"""
    
    def test_plus_empty_interval(self):
        """Test plus operation with empty intervals"""
        I = interval.empty(1)
        v = 1
        I_plus = I + v
        assert I_plus.representsa_('emptySet')
        
        # Reverse operation
        I_plus = v + I
        assert I_plus.representsa_('emptySet')
    
    def test_plus_bounded_interval_numeric(self):
        """Test plus operation between bounded interval and numeric"""
        tol = 1e-9
        
        v = np.array([2, 1])
        I = interval([-2, -1], [3, 4])
        I_plus = v + I
        I_true = interval([0, 0], [5, 5])
        
        assert I_plus == I_true
        
        # Reverse operation
        I_plus = I + v
        assert I_plus == I_true
    
    def test_plus_unbounded_interval_numeric(self):
        """Test plus operation with unbounded intervals"""
        tol = 1e-9
        
        I = interval([-np.inf], [2])
        v = 1
        I_plus = I + v
        I_true = interval([-np.inf], [3])
        
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
        
        # Reverse operation
        I_plus = v + I
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
    
    def test_plus_bounded_intervals(self):
        """Test plus operation between two bounded intervals"""
        tol = 1e-9
        
        I1 = interval([-2, -1], [3, 4])
        I2 = interval([-1, -3], [1, -1])
        I_plus = I1 + I2
        I_true = interval([-3, -4], [4, 3])
        
        assert I_plus == I_true
        
        # Reverse operation (should be commutative)
        I_plus = I2 + I1
        assert I_plus == I_true
    
    def test_plus_unbounded_intervals(self):
        """Test plus operation between unbounded intervals"""
        tol = 1e-9
        
        I1 = interval([-np.inf, -2], [2, 4])
        I2 = interval([-1, 0], [1, np.inf])
        I_plus = I1 + I2
        I_true = interval([-np.inf, -2], [3, np.inf])
        
        # Check bounds
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.inf[1], -2, atol=tol)
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
        assert np.isinf(I_plus.sup[1]) and I_plus.sup[1] > 0
        
        # Reverse operation
        I_plus = I2 + I1
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.inf[1], -2, atol=tol)
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
        assert np.isinf(I_plus.sup[1]) and I_plus.sup[1] > 0
    
    def test_plus_interval_matrix_numeric(self):
        """Test plus operation between interval matrix and numeric"""
        tol = 1e-9
        
        I = interval([[-2, -1], [0, 2]], [[3, 5], [2, 3]])
        v = 2
        I_plus = I + v
        I_true = interval([[0, 1], [2, 4]], [[5, 7], [4, 5]])
        
        assert I_plus == I_true
        
        # Reverse operation
        I_plus = v + I
        assert I_plus == I_true
    
    def test_plus_scalar_operations(self):
        """Test plus operation with scalar values"""
        I = interval([-1, 0], [2, 3])
        
        # Add scalar
        I_plus = I + 5
        expected = interval([4, 5], [7, 8])
        assert I_plus == expected
        
        # Reverse add scalar
        I_plus = 5 + I
        assert I_plus == expected
    
    def test_plus_vector_operations(self):
        """Test plus operation with vector values"""
        I = interval([-1, 0], [2, 3])
        v = np.array([1, -1])
        
        I_plus = I + v
        expected = interval([0, -1], [3, 2])
        assert I_plus == expected
        
        # Reverse operation
        I_plus = v + I
        assert I_plus == expected
    
    def test_plus_matrix_operations(self):
        """Test plus operation with matrix intervals"""
        I1 = interval([[-1, 0], [1, -1]], [[2, 1], [3, 2]])
        I2 = interval([[0, -1], [-1, 0]], [[1, 0], [2, 1]])
        
        I_plus = I1 + I2
        expected = interval([[-1, -1], [0, -1]], [[3, 1], [5, 3]])
        assert I_plus == expected
    
    def test_plus_zero_operations(self):
        """Test plus operation with zero"""
        I = interval([-1, 0], [2, 3])
        
        # Add zero
        I_plus = I + 0
        assert I_plus == I
        
        # Add zero vector
        zero_vec = np.zeros(2)
        I_plus = I + zero_vec
        assert I_plus == I
    
    def test_plus_commutativity(self):
        """Test that plus operation is commutative"""
        I1 = interval([-2, -1], [3, 4])
        I2 = interval([1, 0], [2, 1])
        
        result1 = I1 + I2
        result2 = I2 + I1
        assert result1 == result2
        
        # With numeric
        v = np.array([1, -1])
        result1 = I1 + v
        result2 = v + I1
        assert result1 == result2
    
    def test_plus_associativity(self):
        """Test associativity of plus operation"""
        I1 = interval([-1, 0], [1, 2])
        I2 = interval([0, -1], [2, 1])
        I3 = interval([-1, -1], [1, 1])
        
        # (I1 + I2) + I3
        result1 = (I1 + I2) + I3
        
        # I1 + (I2 + I3)
        result2 = I1 + (I2 + I3)
        
        assert result1 == result2


if __name__ == '__main__':
    pytest.main([__file__]) 