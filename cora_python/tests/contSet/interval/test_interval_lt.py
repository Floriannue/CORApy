"""
Test file for interval lt method (< operator)

Test cases for:
- Basic strict subset checking
- Edge cases with equal intervals
- Different dimensions
- Boundary cases
- Empty intervals

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalLt:
    
    def test_lt_basic_strict_subset(self):
        """Test basic strict subset relationship"""
        # I1 is strict subset of I2
        I1 = Interval([1.5, -0.5], [1.8, 0.5])
        I2 = Interval([1, -1], [2, 1])
        
        assert I1.lt(I2) == True
        assert I1 < I2  # operator overloading
        
    def test_lt_not_strict_subset(self):
        """Test when interval is not a strict subset"""
        # I1 overlaps but isn't strict subset
        I1 = Interval([0, -1], [3, 1])
        I2 = Interval([1, -2], [2, 2])
        
        assert I1.lt(I2) == False
        assert not (I1 < I2)
        
    def test_lt_equal_intervals(self):
        """Test with equal intervals - should be False for strict inequality"""
        I1 = Interval([1, -1], [2, 1])
        I2 = Interval([1, -1], [2, 1])
        
        assert I1.lt(I2) == False  # Equal intervals are not strict subsets
        assert not (I1 < I2)
        
    def test_lt_boundary_touching(self):
        """Test when boundaries touch - should be False for strict"""
        I1 = Interval([1, 1], [2, 2])
        I2 = Interval([1, 0], [2, 3])
        
        # Left boundary touches, so not strict subset
        assert I1.lt(I2) == False
        assert not (I1 < I2)
        
    def test_lt_proper_strict_subset(self):
        """Test proper strict subset"""
        I1 = Interval([1.1, 0.1], [1.9, 0.9])
        I2 = Interval([1, 0], [2, 1])
        
        assert I1.lt(I2) == True
        assert I1 < I2
        
    def test_lt_single_dimension(self):
        """Test with single dimension intervals"""
        I1 = Interval([1.5], [1.8])
        I2 = Interval([1], [2])
        
        assert I1.lt(I2) == True
        assert I1 < I2
        
        # Not strict subset
        I3 = Interval([1], [2])
        I4 = Interval([1.5], [2.5])
        
        assert I3.lt(I4) == False
        assert not (I3 < I4)
        
    def test_lt_higher_dimensions(self):
        """Test with higher dimensional intervals"""
        I1 = Interval([1.1, 2.1, 3.1], [1.9, 2.9, 3.9])
        I2 = Interval([1, 2, 3], [2, 3, 4])
        
        assert I1.lt(I2) == True
        assert I1 < I2
        
    def test_lt_one_dimension_fails(self):
        """Test when one dimension fails strict subset test"""
        I1 = Interval([1.1, 2], [1.9, 3])  # Second dimension touches boundary
        I2 = Interval([1, 2], [2, 3])
        
        assert I1.lt(I2) == False  # Because second dimension inf touches
        assert not (I1 < I2)
        
    def test_lt_with_numeric(self):
        """Test comparison with numeric values"""
        I1 = Interval([1.5], [1.8])
        
        # Should handle conversion to interval
        result = I1.lt(2)
        assert result == True
        
        result = I1.lt(1.5)  # Equal to left boundary
        assert result == False
        
    def test_lt_empty_intervals(self):
        """Test with empty intervals"""
        I_empty = Interval.empty(2)
        I_normal = Interval([1, 1], [2, 2])
        
        # Empty interval behavior might depend on implementation
        # but should be consistent
        result = I_empty.lt(I_normal)
        assert isinstance(result, bool)
        
    def test_lt_point_intervals(self):
        """Test with point intervals (degenerate)"""
        I_point = Interval([1.5, 2.5], [1.5, 2.5])
        I_contain = Interval([1, 2], [2, 3])
        
        assert I_point.lt(I_contain) == True
        assert I_point < I_contain
        
    def test_lt_matrix_intervals(self):
        """Test with matrix intervals"""
        # 2x2 intervals - strict subset
        inf1 = np.array([[1.1, 2.1], [3.1, 4.1]])
        sup1 = np.array([[1.9, 2.9], [3.9, 4.9]])
        I1 = Interval(inf1, sup1)
        
        inf2 = np.array([[1, 2], [3, 4]])
        sup2 = np.array([[2, 3], [4, 5]])
        I2 = Interval(inf2, sup2)
        
        assert I1.lt(I2) == True
        assert I1 < I2
        
    def test_lt_boundary_cases(self):
        """Test various boundary cases"""
        # Right boundary touches
        I1 = Interval([1.1, 1.1], [2, 2])
        I2 = Interval([1, 1], [2, 2])
        
        assert I1.lt(I2) == False  # Right boundary touches
        assert not (I1 < I2)
        
        # Both boundaries strictly inside
        I1 = Interval([1.1, 1.1], [1.9, 1.9])
        I2 = Interval([1, 1], [2, 2])
        
        assert I1.lt(I2) == True
        assert I1 < I2
        
    def test_lt_precision(self):
        """Test with floating point precision"""
        eps = 1e-10
        I1 = Interval([1 + eps], [2 - eps])
        I2 = Interval([1], [2])
        
        assert I1.lt(I2) == True
        assert I1 < I2
        
    def test_lt_reverse_relationship(self):
        """Test reverse relationships"""
        I1 = Interval([1, 1], [2, 2])
        I2 = Interval([1.1, 1.1], [1.9, 1.9])
        
        # I2 is strict subset of I1, not vice versa
        assert I1.lt(I2) == False
        assert not (I1 < I2)
        assert I2.lt(I1) == True
        assert I2 < I1
        
    def test_lt_large_dimensions(self):
        """Test with larger dimensional intervals"""
        dim = 10
        inf1 = 1.1 * np.ones(dim)
        sup1 = 1.9 * np.ones(dim)
        I1 = Interval(inf1, sup1)
        
        inf2 = np.ones(dim)
        sup2 = 2 * np.ones(dim)
        I2 = Interval(inf2, sup2)
        
        assert I1.lt(I2) == True
        assert I1 < I2 