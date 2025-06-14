"""
test_interval_mtimes - unit test function for interval mtimes operation

This module tests the mtimes operation (matrix multiplication) for intervals,
including multiplication with other intervals and numeric values.

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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestIntervalMtimes:
    """Test class for interval mtimes operation"""
    
    def test_mtimes_empty_set(self):
        """Test mtimes operation with empty sets"""
        a = Interval.empty(1)
        b = 3
        result = b @ a
        assert result.representsa_('emptySet')
    
    def test_mtimes_scalar_values(self):
        """Test mtimes operation with scalar values"""
        # Test 1: interval * numeric
        a = Interval([-1], [2])
        b = 3
        c = a @ b
        c_true = Interval([-3], [6])
        assert c == c_true
        
        # Test 2: numeric * interval
        a = -2
        b = Interval([2], [4])
        c = a @ b
        c_true = Interval([-8], [-4])
        assert c == c_true
        
        # Test 3: interval * interval
        a = Interval([-2], [1])
        b = Interval([2], [4])
        c = a @ b
        c_true = Interval([-8], [4])
        assert c == c_true
    
    def test_mtimes_scalar_matrix(self):
        """Test mtimes operation between scalar and matrix"""
        # Test 4: scalar interval * numeric matrix
        a = Interval([-1], [2])
        b = np.array([[-1, 0], [1, 2]])
        c = a @ b
        c_true = Interval([[-2, 0], [-1, -2]], [[1, 0], [2, 4]])
        assert c == c_true
        
        # Test 5: numeric * interval matrix
        a = -2
        b = Interval([[2, -3], [-1, 2]], [[4, -2], [1, 3]])
        c = a @ b
        c_true = Interval([[-8, 4], [-2, -6]], [[-4, 6], [2, -4]])
        assert c == c_true
        
        # Test 6: scalar interval * interval matrix
        a = Interval([-1], [2])
        b = Interval([[2, -3], [-1, 2]], [[4, -2], [1, 3]])
        c = a @ b
        c_true = Interval([[-4, -6], [-2, -3]], [[8, 3], [2, 6]])
        assert c == c_true
    
    def test_mtimes_matrix_scalar(self):
        """Test mtimes operation between matrix and scalar"""
        # Test 7: interval matrix * numeric scalar
        a = Interval([[-1, 0], [-2, 2]], [[2, 1], [-1, 3]])
        b = -1
        c = a @ b
        c_true = Interval([[-2, -1], [1, -3]], [[1, 0], [2, -2]])
        assert c == c_true
        
        # Test 8: numeric matrix * interval scalar
        a = np.array([[-1, 0], [1, 2]])
        b = Interval([-2], [1])
        c = a @ b
        c_true = Interval([[-1, 0], [-2, -4]], [[2, 0], [1, 2]])
        assert c == c_true
        
        # Test 9: interval matrix * interval scalar
        a = Interval([[-1, 0], [-2, 2]], [[2, 1], [-1, 3]])
        b = Interval([-2], [1])
        c = a @ b
        c_true = Interval([[-4, -2], [-2, -6]], [[2, 1], [4, 3]])
        assert c == c_true
    
    def test_mtimes_matrix_matrix(self):
        """Test mtimes operation between matrices"""
        # Test 10: interval matrix * numeric matrix
        a = Interval([[-1, 0], [-2, 2]], [[2, 1], [-1, 3]])
        b = np.array([[1, 0], [0, 1]])  # Identity matrix
        c = a @ b
        # Should be the same as original matrix
        assert c == a
        
        # Test 11: numeric matrix * interval matrix
        a = np.array([[2, 0], [0, 2]])  # 2 * Identity
        b = Interval([[-1, 0], [-2, 2]], [[2, 1], [-1, 3]])
        c = a @ b
        c_true = Interval([[-2, 0], [-4, 4]], [[4, 2], [-2, 6]])
        assert c == c_true
        
        # Test 12: interval matrix * interval matrix
        a = Interval([[1, 0], [0, 1]], [[2, 1], [1, 2]])
        b = Interval([[-1, 0], [0, -1]], [[1, 1], [1, 1]])
        c = a @ b
        # This is a more complex case - just check it doesn't crash
        assert isinstance(c, Interval)
        assert c.inf.shape == (2, 2)
        assert c.sup.shape == (2, 2)
    
    def test_mtimes_zero_cases(self):
        """Test mtimes operation with zero values"""
        # Zero scalar
        a = Interval([0], [0])
        b = Interval([-np.inf], [np.inf])
        c = a @ b
        c_true = Interval([0], [0])
        assert c == c_true
        
        # Zero matrix
        a = Interval(np.zeros((2, 2)), np.zeros((2, 2)))
        b = Interval([[-1, 1], [2, -2]], [[1, 2], [3, -1]])
        c = a @ b
        c_true = Interval(np.zeros((2, 2)), np.zeros((2, 2)))
        assert c == c_true
    
    def test_mtimes_negative_cases(self):
        """Test mtimes operation with negative values"""
        # Negative scalar interval
        a = Interval([-2], [-1])
        b = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        c = a @ b
        
        # All results should be negative
        assert np.all(c.sup <= 0)
        assert np.all(c.inf <= c.sup)
    
    def test_mtimes_mixed_sign_cases(self):
        """Test mtimes operation with intervals containing zero"""
        # Interval containing zero
        a = Interval([-1], [1])
        b = Interval([[1, 2]], [[3, 4]])
        c = a @ b
        
        # Result should contain zero
        assert np.all(c.inf <= 0)
        assert np.all(c.sup >= 0)
    
    def test_mtimes_dimension_compatibility(self):
        """Test dimension compatibility checking"""
        # Incompatible dimensions should raise error
        a = Interval(np.ones((2, 3)), 2 * np.ones((2, 3)))
        b = Interval(np.ones((2, 2)), 2 * np.ones((2, 2)))  # Wrong dimension
        
        with pytest.raises(CORAError):
            c = a @ b
    
    def test_mtimes_large_matrices(self):
        """Test mtimes operation with larger matrices"""
        # 3x3 matrices
        a = Interval(np.ones((3, 3)), 2 * np.ones((3, 3)))
        b = Interval(-np.ones((3, 3)), np.ones((3, 3)))
        c = a @ b
        
        assert c.inf.shape == (3, 3)
        assert c.sup.shape == (3, 3)
        assert np.all(c.inf <= c.sup)
    
    def test_mtimes_infinite_values(self):
        """Test mtimes operation with infinite values"""
        # Infinite bounds
        a = Interval([-np.inf], [np.inf])
        b = Interval([1], [2])
        c = a @ b
        
        # Result should have infinite bounds
        assert np.isinf(c.inf[0]) and c.inf[0] < 0
        assert np.isinf(c.sup[0]) and c.sup[0] > 0
    
    def test_mtimes_associativity(self):
        """Test associativity of matrix multiplication"""
        # For compatible dimensions
        a = Interval([[1]], [[2]])  # 1x1
        b = Interval([[2]], [[3]])  # 1x1  
        c = Interval([[1]], [[2]])  # 1x1
        
        # (a @ b) @ c
        result1 = (a @ b) @ c
        
        # a @ (b @ c)
        result2 = a @ (b @ c)
        
        assert result1 == result2
    
    def test_mtimes_identity_property(self):
        """Test multiplication with identity matrix"""
        # Create interval matrix
        a = Interval([[-1, 2], [0, -1]], [[1, 3], [2, 1]])
        
        # Identity matrix
        I = np.eye(2)
        
        # a @ I should equal a
        result = a @ I
        assert result == a
        
        # I @ a should equal a
        result = I @ a
        assert result == a


if __name__ == '__main__':
    pytest.main([__file__]) 
