import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalTril:
    def test_tril_default(self):
        I = Interval(-np.ones((3, 3)), np.ones((3, 3)))
        res = I.tril()
        expected = Interval(np.tril(-np.ones((3, 3))), np.tril(np.ones((3, 3))))
        assert res.isequal(expected)

    def test_tril_with_k(self):
        I = Interval(-np.ones((3, 3)), np.ones((3, 3)))
        res = I.tril(k=1)
        expected = Interval(np.tril(-np.ones((3, 3)), k=1), np.tril(np.ones((3, 3)), k=1))
        assert res.isequal(expected)

    def test_tril_empty(self):
        I = Interval.empty(2)
        res = I.tril()
        assert res.is_empty() 