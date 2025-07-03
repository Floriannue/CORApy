import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalIsscalar:
    def test_isscalar_scalar(self):
        I = Interval(1, 2)
        assert I.isscalar()

    def test_isscalar_vector(self):
        I = Interval([1, 2], [3, 4])
        assert not I.isscalar()

    def test_isscalar_matrix(self):
        I = Interval(np.ones((2, 2)), 2 * np.ones((2, 2)))
        assert not I.isscalar()
        
    def test_isscalar_empty(self):
        I = Interval.empty()
        assert not I.isscalar() 