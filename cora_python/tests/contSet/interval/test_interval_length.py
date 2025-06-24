"""
Test file for interval length method

Authors: Matthias Althoff (MATLAB), Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalLength:
    
    def test_length_basic(self):
        """Test basic length functionality"""
        I = Interval([[-1, 1]], [[1, 2]])  # 1x2 matrix
        
        result = I.length()
        assert result == 2  # longest dimension
        
    def test_length_vector(self):
        """Test length with vector intervals"""
        I = Interval([1, 2, 3], [4, 5, 6])  # 3D vector
        
        result = I.length()
        assert result == 3
        
    def test_length_matrix(self):
        """Test length with matrix intervals"""
        inf = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
        sup = np.array([[2, 3], [4, 5], [6, 7]])
        I = Interval(inf, sup)
        
        result = I.length()
        assert result == 3  # max(3, 2) = 3
        
    def test_length_scalar(self):
        """Test length with scalar interval"""
        I = Interval([1], [2])
        
        result = I.length()
        assert result == 1
        
    def test_length_square_matrix(self):
        """Test length with square matrix"""
        inf = np.ones((5, 5))
        sup = 2 * np.ones((5, 5))
        I = Interval(inf, sup)
        
        result = I.length()
        assert result == 5 