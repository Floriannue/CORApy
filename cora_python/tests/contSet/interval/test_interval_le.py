"""
Test file for interval le method (<= operator)

Test cases for:
- Basic subset checking
- Edge cases with equal intervals
- Different dimensions
- Numeric comparisons
- Empty intervals

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalLe:
    
    def test_le_basic_subset(self):
        """Test basic subset relationship"""
        # I1 is subset of I2
        I1 = Interval([1, -1], [2, 1])
        I2 = Interval([1, -2], [2, 2])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True  # operator overloading
        
    def test_le_not_subset(self):
        """Test when interval is not a subset"""
        # I1 is not subset of I2
        I1 = Interval([0, -1], [3, 1])
        I2 = Interval([1, -2], [2, 2])
        
        assert I1.le(I2) == False
        assert (I1 <= I2) == False
        
    def test_le_equal_intervals(self):
        """Test with equal intervals"""
        I1 = Interval([1, -1], [2, 1])
        I2 = Interval([1, -1], [2, 1])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
    def test_le_single_dimension(self):
        """Test with single dimension intervals"""
        I1 = Interval([1], [2])
        I2 = Interval([0], [3])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
        I3 = Interval([0], [3])
        I4 = Interval([1], [2])
        
        assert I3.le(I4) == False
        assert (I3 <= I4) == False
        
    def test_le_higher_dimensions(self):
        """Test with higher dimensional intervals"""
        I1 = Interval([1, 2, 3], [2, 3, 4])
        I2 = Interval([0, 1, 2], [3, 4, 5])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
    def test_le_partial_overlap(self):
        """Test intervals with partial overlap"""
        I1 = Interval([1, 1], [3, 3])
        I2 = Interval([2, 0], [4, 2])
        
        assert I1.le(I2) == False
        assert (I1 <= I2) == False
        
    def test_le_boundary_cases(self):
        """Test boundary cases"""
        # Touching boundaries
        I1 = Interval([1, 1], [2, 2])
        I2 = Interval([1, 1], [2, 2])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
        # One boundary touches
        I1 = Interval([1, 1], [2, 2])
        I2 = Interval([1, 0], [2, 3])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
    def test_le_with_numeric(self):
        """Test comparison with numeric values"""
        I1 = Interval([1], [1])  # degenerate interval
        
        # Should handle conversion to interval
        result = I1.le(1)
        assert result == True
        
    def test_le_empty_intervals(self):
        """Test with empty intervals"""
        I_empty = Interval.empty(2)
        I_normal = Interval([0, 0], [1, 1])
        
        # Empty interval should be subset of any interval
        assert I_empty.le(I_normal) == True
        
    def test_le_point_intervals(self):
        """Test with point intervals (degenerate)"""
        I_point = Interval([1, 2], [1, 2])
        I_contain = Interval([0, 1], [2, 3])
        
        assert I_point.le(I_contain) == True
        assert (I_point <= I_contain) == True
        
    def test_le_matrix_intervals(self):
        """Test with matrix intervals"""
        # 2x2 intervals
        inf = np.array([[1, 2], [3, 4]])
        sup = np.array([[2, 3], [4, 5]])
        I1 = Interval(inf, sup)
        
        inf2 = np.array([[0, 1], [2, 3]])
        sup2 = np.array([[3, 4], [5, 6]])
        I2 = Interval(inf2, sup2)
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
    def test_le_precision(self):
        """Test with floating point precision issues"""
        I1 = Interval([1.0000001], [1.9999999])
        I2 = Interval([1.0], [2.0])
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
    def test_le_large_dimensions(self):
        """Test with larger dimensional intervals"""
        dim = 10
        inf1 = np.ones(dim)
        sup1 = 2 * np.ones(dim)
        I1 = Interval(inf1, sup1)
        
        inf2 = np.zeros(dim)
        sup2 = 3 * np.ones(dim)
        I2 = Interval(inf2, sup2)
        
        assert I1.le(I2) == True
        assert (I1 <= I2) == True
        
    def test_le_strict_inequality(self):
        """Test cases where <= should be true but < would be false"""
        I1 = Interval([1, 1], [2, 2])
        I2 = Interval([1, 1], [2, 2])  # Equal intervals
        
        assert I1.le(I2) == True  # <= should be True
        assert (I1 <= I2) == True 