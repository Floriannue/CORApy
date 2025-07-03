import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalTriu:
    def test_triu_default(self):
        I = Interval(-np.ones((2,2)), np.ones((2,2)))
        res = I.triu()
        expected = Interval([[-1, -1], [0, -1]], [[1, 1], [0, 1]])
        assert res.isequal(expected)

    def test_triu_k1(self):
        I = Interval(-np.ones((2,2)), np.ones((2,2)))
        res = I.triu(k=1)
        expected = Interval([[0, -1], [0, 0]], [[0, 1], [0, 0]])
        assert res.isequal(expected)
        
    def test_triu_empty(self):
        I = Interval.empty(2)
        res = I.triu()
        assert res.is_empty() 