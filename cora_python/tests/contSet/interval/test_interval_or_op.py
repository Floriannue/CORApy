"""
Test file for interval or_op method (union operation)

Test cases for:
- Basic union operations
- Union with intervals
- Union with numeric values
- Empty interval unions
- Different dimensions
- Edge cases

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalOrOp:
    
    def test_or_op_basic_union(self):
        """Test basic union of two intervals"""
        I1 = Interval([-2, -2], [-1, -1])
        I2 = Interval([0, 0], [2, 2])
        
        result = I1.or_op(I2)
        assert isinstance(result, Interval)
        
        # Union should contain both intervals
        expected_inf = np.array([-2, -2])
        expected_sup = np.array([2, 2])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_operator_overload(self):
        """Test | operator overloading"""
        I1 = Interval([-1, -1], [0, 0])
        I2 = Interval([1, 1], [2, 2])
        
        result = I1 | I2
        assert isinstance(result, Interval)
        
        expected_inf = np.array([-1, -1])
        expected_sup = np.array([2, 2])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_overlapping_intervals(self):
        """Test union of overlapping intervals"""
        I1 = Interval([0, 0], [2, 2])
        I2 = Interval([1, 1], [3, 3])
        
        result = I1.or_op(I2)
        
        expected_inf = np.array([0, 0])
        expected_sup = np.array([3, 3])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_identical_intervals(self):
        """Test union of identical intervals"""
        I1 = Interval([1, 2], [3, 4])
        I2 = Interval([1, 2], [3, 4])
        
        result = I1.or_op(I2)
        
        np.testing.assert_array_equal(result.inf, I1.inf)
        np.testing.assert_array_equal(result.sup, I1.sup)
        
    def test_or_op_with_numeric(self):
        """Test union with numeric values"""
        I = Interval([1, 1], [2, 2])
        point = np.array([0, 3])
        
        result = I.or_op(point)
        
        expected_inf = np.array([0, 1])
        expected_sup = np.array([2, 3])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_with_scalar(self):
        """Test union with scalar value"""
        I = Interval([1], [2])
        scalar = 3
        
        result = I.or_op(scalar)
        
        expected_inf = np.array([1])
        expected_sup = np.array([3])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_empty_interval(self):
        """Test union with empty intervals"""
        I_normal = Interval([1, 1], [2, 2])
        I_empty = Interval.empty(2)
        
        # Union with empty interval should return the normal interval
        result1 = I_normal.or_op(I_empty)
        np.testing.assert_array_equal(result1.inf, I_normal.inf)
        np.testing.assert_array_equal(result1.sup, I_normal.sup)
        
        # Reverse order
        result2 = I_empty.or_op(I_normal)
        np.testing.assert_array_equal(result2.inf, I_normal.inf)
        np.testing.assert_array_equal(result2.sup, I_normal.sup)
        
    def test_or_op_single_dimension(self):
        """Test union with single dimension"""
        I1 = Interval([1], [2])
        I2 = Interval([3], [4])
        
        result = I1.or_op(I2)
        
        expected_inf = np.array([1])
        expected_sup = np.array([4])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_higher_dimensions(self):
        """Test union with higher dimensions"""
        I1 = Interval([1, 2, 3], [2, 3, 4])
        I2 = Interval([0, 1, 5], [1, 2, 6])
        
        result = I1.or_op(I2)
        
        expected_inf = np.array([0, 1, 3])
        expected_sup = np.array([2, 3, 6])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_negative_values(self):
        """Test union with negative values"""
        I1 = Interval([-3, -2], [-1, 0])
        I2 = Interval([-5, -4], [-4, -3])
        
        result = I1.or_op(I2)
        
        expected_inf = np.array([-5, -4])
        expected_sup = np.array([-1, 0])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_point_intervals(self):
        """Test union with point intervals (degenerate)"""
        I_point1 = Interval([1, 2], [1, 2])
        I_point2 = Interval([3, 4], [3, 4])
        
        result = I_point1.or_op(I_point2)
        
        expected_inf = np.array([1, 2])
        expected_sup = np.array([3, 4])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_subset_case(self):
        """Test union where one interval is subset of another"""
        I_small = Interval([1.5, 1.5], [2.5, 2.5])
        I_large = Interval([1, 1], [3, 3])
        
        result = I_small.or_op(I_large)
        
        # Result should be the larger interval
        np.testing.assert_array_equal(result.inf, I_large.inf)
        np.testing.assert_array_equal(result.sup, I_large.sup)
        
    def test_or_op_matrix_intervals(self):
        """Test union with matrix intervals"""
        inf1 = np.array([[1, 2], [3, 4]])
        sup1 = np.array([[2, 3], [4, 5]])
        I1 = Interval(inf1, sup1)
        
        inf2 = np.array([[0, 1], [2, 6]])
        sup2 = np.array([[1, 2], [3, 7]])
        I2 = Interval(inf2, sup2)
        
        result = I1.or_op(I2)
        
        expected_inf = np.array([[0, 1], [2, 4]])
        expected_sup = np.array([[2, 3], [4, 7]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_or_op_different_modes(self):
        """Test union with different modes"""
        I1 = Interval([1, 1], [2, 2])
        I2 = Interval([3, 3], [4, 4])
        
        # Test different modes (though implementation may be the same)
        result_outer = I1.or_op(I2, 'outer')
        result_exact = I1.or_op(I2, 'exact')
        
        expected_inf = np.array([1, 1])
        expected_sup = np.array([4, 4])
        
        np.testing.assert_array_equal(result_outer.inf, expected_inf)
        np.testing.assert_array_equal(result_outer.sup, expected_sup)
        np.testing.assert_array_equal(result_exact.inf, expected_inf)
        np.testing.assert_array_equal(result_exact.sup, expected_sup)
        
    def test_or_op_precision(self):
        """Test union with floating point precision"""
        I1 = Interval([1.0000001], [1.9999999])
        I2 = Interval([2.0000001], [2.9999999])
        
        result = I1.or_op(I2)
        
        expected_inf = np.array([1.0000001])
        expected_sup = np.array([2.9999999])
        
        np.testing.assert_array_almost_equal(result.inf, expected_inf)
        np.testing.assert_array_almost_equal(result.sup, expected_sup)
        
    def test_or_op_large_dimensions(self):
        """Test union with larger dimensions"""
        dim = 10
        I1 = Interval(np.ones(dim), 2 * np.ones(dim))
        I2 = Interval(3 * np.ones(dim), 4 * np.ones(dim))
        
        result = I1.or_op(I2)
        
        expected_inf = np.ones(dim)
        expected_sup = 4 * np.ones(dim)
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup) 