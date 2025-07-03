import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalSum:
    def test_sum_default(self):
        I = Interval(np.array([[-3, 4], [-2, 1], [-4, 6]]), np.array([[-2, 5], [-1, 2], [-3, 7]]))
        res = I.sum()
        expected = Interval([-9, 11], [-6, 14])
        assert res.isequal(expected)

    def test_sum_axis0(self):
        I = Interval(np.array([[-3, 4], [-2, 1], [-4, 6]]), np.array([[-2, 5], [-1, 2], [-3, 7]]))
        res = I.sum(axis=0)
        expected = Interval([-9, 11], [-6, 14])
        assert res.isequal(expected)

    def test_sum_axis1(self):
        I = Interval(np.array([[-3, 4], [-2, 1], [-4, 6]]), np.array([[-2, 5], [-1, 2], [-3, 7]]))
        res = I.sum(axis=1)
        expected = Interval([1, -1, 2], [3, 1, 4])
        assert res.isequal(expected)

    def test_sum_vector(self):
        I = Interval([-3, -2, -4], [4, 1, 6])
        res = I.sum()
        expected = Interval(-9, 11)
        assert res.isequal(expected)
        
    def test_sum_empty(self):
        I = Interval.empty(2)
        res = I.sum()
        assert res.is_empty() 