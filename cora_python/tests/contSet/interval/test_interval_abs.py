import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import Interval

class TestIntervalAbs:
    """Test class for interval abs operation"""

    def test_abs_positive_interval(self):
        """Test abs on an interval with only positive values"""
        I = Interval([1, 2], [3, 4])
        res = abs(I)
        expected = Interval([1, 2], [3, 4])
        assert res == expected

    def test_abs_negative_interval(self):
        """Test abs on an interval with only negative values"""
        I = Interval([-3, -4], [-1, -2])
        res = abs(I)
        expected = Interval([1, 2], [3, 4])
        assert res == expected

    def test_abs_mixed_sign_interval(self):
        """Test abs on an interval containing zero"""
        I = Interval([-2, -3], [1, 4])
        res = abs(I)
        expected = Interval([0, 0], [2, 4])
        assert res == expected

    def test_abs_zero_bound_interval(self):
        """Test abs on intervals with zero as a bound"""
        I1 = Interval([0, -5], [5, -1])
        res1 = abs(I1)
        expected1 = Interval([0, 1], [5, 5])
        assert res1 == expected1

        I2 = Interval([-5, 0], [-1, 6])
        res2 = abs(I2)
        expected2 = Interval([1, 0], [5, 6])
        assert res2 == expected2

    def test_abs_point_interval(self):
        """Test abs on a point interval"""
        I1 = Interval([-2, -2])
        res1 = abs(I1)
        expected1 = Interval([2, 2])
        assert res1 == expected1

        I2 = Interval([3, 3])
        res2 = abs(I2)
        expected2 = Interval([3, 3])
        assert res2 == expected2
        
        I3 = Interval([0, 0])
        res3 = abs(I3)
        expected3 = Interval([0, 0])
        assert res3 == expected3

    def test_abs_multi_dimensional_interval(self):
        """Test abs on a vector of intervals"""
        inf = np.array([-2, 1, -5])
        sup = np.array([3, 4, -1])
        I = Interval(inf, sup)
        res = abs(I)
        
        expected_inf = np.array([0, 1, 1])
        expected_sup = np.array([3, 4, 5])
        expected = Interval(expected_inf, expected_sup)
        assert res == expected

    def test_abs_interval_matrix(self):
        """Test abs on an interval matrix"""
        inf = np.array([[-2, 1], [-5, 0]])
        sup = np.array([[3, 4], [-1, 6]])
        I = Interval(inf, sup)
        res = abs(I)

        expected_inf = np.array([[0, 1], [1, 0]])
        expected_sup = np.array([[3, 4], [5, 6]])
        expected = Interval(expected_inf, expected_sup)
        assert res == expected

    def test_abs_unbounded_interval(self):
        """Test abs on an unbounded interval"""
        I1 = Interval([-np.inf], [2])
        res1 = abs(I1)
        expected1 = Interval([0], [np.inf])
        assert res1.inf[0] == 0
        assert np.isinf(res1.sup[0]) and res1.sup[0] > 0
        
        I2 = Interval([-np.inf], [-2])
        res2 = abs(I2)
        expected2 = Interval([2], [np.inf])
        assert res2.inf[0] == 2
        assert np.isinf(res2.sup[0]) and res2.sup[0] > 0

        I3 = Interval([2], [np.inf])
        res3 = abs(I3)
        expected3 = Interval([2], [np.inf])
        assert res3.inf[0] == 2
        assert np.isinf(res3.sup[0]) and res3.sup[0] > 0

    def test_abs_empty_interval(self):
        """Test abs on an empty interval"""
        I = Interval.empty(2)
        res = abs(I)
        assert res.is_empty() 