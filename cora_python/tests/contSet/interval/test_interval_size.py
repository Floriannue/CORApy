"""
Test file for interval size method

Authors: Matthias Althoff (MATLAB), Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalSize:
    
    def test_size_basic(self):
        """Test basic size functionality"""
        I = Interval([[-1, 1]], [[1, 2]])  # 1x2 matrix
        
        result = I.size()
        assert result == (1, 2)
        
    def test_size_vector(self):
        """Test size with vector intervals"""
        I = Interval([1, 2, 3], [4, 5, 6])  # 3D vector
        
        result = I.size()
        assert result == (3,)
        
    def test_size_matrix(self):
        """Test size with matrix intervals"""
        inf = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
        sup = np.array([[2, 3], [4, 5], [6, 7]])
        I = Interval(inf, sup)
        
        result = I.size()
        assert result == (3, 2)
        
    def test_size_with_dimension(self):
        """Test size with specific dimension"""
        inf = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
        sup = np.array([[2, 3], [4, 5], [6, 7]])
        I = Interval(inf, sup)
        
        assert I.size(0) == 3  # rows
        assert I.size(1) == 2  # columns
        
    def test_size_scalar(self):
        """Test size with scalar interval"""
        I = Interval([1], [2])
        
        result = I.size()
        assert result == (1,)
        
    def test_size_dimension_out_of_bounds(self):
        """Test size with dimension index out of bounds"""
        I = Interval([[1, 2]], [[3, 4]])  # 1x2 matrix
        
        assert I.size(2) == 1  # Should return 1 for out-of-bounds dimension 