import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalProd:
    def test_prod_default(self):
        I = Interval([[-1, 1], [2, 3]], [[0, 2], [3, 4]])
        res = I.prod()
        expected = Interval([-3, 3], [0, 8])
        assert res.isequal(expected)

    def test_prod_axis0(self):
        I = Interval([[-1, 1], [2, 3]], [[0, 2], [3, 4]])
        res = I.prod(axis=0)
        expected = Interval([-3, 3], [0, 8])
        assert res.isequal(expected)

    def test_prod_axis1(self):
        I = Interval([[-1, 1], [2, 3]], [[0, 2], [3, 4]])
        res = I.prod(axis=1)
        expected = Interval([-2, 6], [0, 12])
        assert res.isequal(expected)

    def test_prod_vector(self):
        I = Interval([-1, 2, -3], [0, 3, -2])
        res = I.prod()
        expected = Interval(0, 9)
        assert res.isequal(expected)
        
    def test_prod_empty(self):
        I = Interval.empty(2)
        res = I.prod()
        assert res.is_empty() 