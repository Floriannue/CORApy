"""
Test file for interval horzcat method (horizontal concatenation)

Test cases for:
- Basic horizontal concatenation
- Multiple intervals
- Mixed interval and numeric types
- Different dimensions
- Edge cases

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalHorzcat:
    
    def test_horzcat_basic(self):
        """Test basic horizontal concatenation"""
        I1 = Interval([-1], [1])
        I2 = Interval([1], [2])
        
        result = I1.horzcat(I2)
        assert isinstance(result, Interval)
        
        expected_inf = np.array([[-1, 1]])
        expected_sup = np.array([[1, 2]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_multiple_intervals(self):
        """Test concatenation of multiple intervals"""
        I1 = Interval([-1], [1])
        I2 = Interval([0], [2])
        I3 = Interval([1], [3])
        
        result = I1.horzcat(I2, I3)
        
        expected_inf = np.array([[-1, 0, 1]])
        expected_sup = np.array([[1, 2, 3]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_with_numeric(self):
        """Test concatenation with numeric values"""
        I = Interval([-1], [1])
        numeric = 2
        
        result = I.horzcat(numeric)
        
        expected_inf = np.array([[-1, 2]])
        expected_sup = np.array([[1, 2]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_mixed_types(self):
        """Test concatenation with mixed interval and numeric types"""
        I1 = Interval([1], [2])
        numeric = 3
        I2 = Interval([4], [5])
        
        result = I1.horzcat(numeric, I2)
        
        expected_inf = np.array([[1, 3, 4]])
        expected_sup = np.array([[2, 3, 5]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_vector_intervals(self):
        """Test concatenation of vector intervals"""
        I1 = Interval([1, 2], [3, 4])  # 2D interval
        I2 = Interval([5, 6], [7, 8])  # 2D interval
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[1, 5], [2, 6]])
        expected_sup = np.array([[3, 7], [4, 8]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_matrix_intervals(self):
        """Test concatenation of matrix intervals"""
        inf1 = np.array([[1, 2]])
        sup1 = np.array([[3, 4]])
        I1 = Interval(inf1, sup1)
        
        inf2 = np.array([[5, 6]])
        sup2 = np.array([[7, 8]])
        I2 = Interval(inf2, sup2)
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[1, 2, 5, 6]])
        expected_sup = np.array([[3, 4, 7, 8]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_empty_list(self):
        """Test with empty argument list"""
        with pytest.raises(ValueError, match="At least one argument required"):
            Interval.horzcat()
            
    def test_horzcat_single_argument(self):
        """Test with single argument"""
        I = Interval([1, 2], [3, 4])
        result = I.horzcat()
        
        np.testing.assert_array_equal(result.inf, I.inf)
        np.testing.assert_array_equal(result.sup, I.sup)
        
    def test_horzcat_non_interval_first(self):
        """Test when first argument is not an interval"""
        numeric = 1
        I = Interval([2], [3])
        
        result = Interval.horzcat(numeric, I)
        
        expected_inf = np.array([[1, 2]])
        expected_sup = np.array([[1, 3]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_point_intervals(self):
        """Test concatenation of point intervals"""
        I1 = Interval([1], [1])  # Point interval
        I2 = Interval([2], [2])  # Point interval
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[1, 2]])
        expected_sup = np.array([[1, 2]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_negative_values(self):
        """Test concatenation with negative values"""
        I1 = Interval([-3], [-1])
        I2 = Interval([-2], [0])
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[-3, -2]])
        expected_sup = np.array([[-1, 0]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_different_vector_sizes(self):
        """Test concatenation with different vector sizes"""
        I1 = Interval([1, 2], [3, 4])  # 2D
        I2 = Interval([5, 6], [7, 8])  # 2D
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[1, 5], [2, 6]])
        expected_sup = np.array([[3, 7], [4, 8]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_with_array(self):
        """Test concatenation with numpy arrays"""
        I = Interval([1, 2], [3, 4])
        arr = np.array([5, 6])
        
        result = I.horzcat(arr)
        
        expected_inf = np.array([[1, 5], [2, 6]])
        expected_sup = np.array([[3, 5], [4, 6]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_precision(self):
        """Test concatenation with floating point precision"""
        I1 = Interval([1.111111], [2.222222])
        I2 = Interval([3.333333], [4.444444])
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[1.111111, 3.333333]])
        expected_sup = np.array([[2.222222, 4.444444]])
        
        np.testing.assert_array_almost_equal(result.inf, expected_inf)
        np.testing.assert_array_almost_equal(result.sup, expected_sup)
        
    def test_horzcat_large_number_of_intervals(self):
        """Test concatenation of large number of intervals"""
        intervals = [Interval([i], [i+1]) for i in range(10)]
        
        result = intervals[0].horzcat(*intervals[1:])
        
        expected_inf = np.array([list(range(10))])
        expected_sup = np.array([list(range(1, 11))])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_horzcat_zero_intervals(self):
        """Test concatenation with zero intervals"""
        I1 = Interval([0], [0])
        I2 = Interval([0], [0])
        
        result = I1.horzcat(I2)
        
        expected_inf = np.array([[0, 0]])
        expected_sup = np.array([[0, 0]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup) 