import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalIsnan:
    def test_isnan(self):
        I = Interval(-np.ones((2,2)), np.ones((2,2)))
        assert not I.isnan()

    def test_isnan_empty(self):
        I = Interval.empty(2)
        assert not I.isnan() 