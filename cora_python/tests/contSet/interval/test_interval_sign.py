import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalSign:
    def test_sign_positive(self):
        I = Interval([2, 5], [3, 6])
        res = I.sign()
        expected = Interval([1, 1], [1, 1])
        assert res.isequal(expected)

    def test_sign_negative(self):
        I = Interval([-5, -2], [-4, -1])
        res = I.sign()
        expected = Interval([-1, -1], [-1, -1])
        assert res.isequal(expected)

    def test_sign_mixed(self):
        I = Interval([-2, 3], [3, 4])
        res = I.sign()
        expected = Interval([-1, 1], [1, 1])
        assert res.isequal(expected)

    def test_sign_containing_zero(self):
        I = Interval(-5, 5)
        res = I.sign()
        expected = Interval(-1, 1)
        assert res.isequal(expected)
        
    def test_sign_at_zero(self):
        I = Interval(0,0)
        res = I.sign()
        expected = Interval(0,0)
        assert res.isequal(expected)

    def test_sign_empty(self):
        I = Interval.empty(2)
        res = I.sign()
        assert res.is_empty()
        assert res.dim() == 2 