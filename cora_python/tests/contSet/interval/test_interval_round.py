import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalRound:
    def test_round_default(self):
        I = Interval([-1.5, 1.1], [1.2, 2.8])
        res = I.round()
        expected = Interval([-2.0, 1.0], [1.0, 3.0])
        assert res.isequal(expected)

    def test_round_with_N(self):
        I = Interval([-1.55, 1.11], [1.22, 2.88])
        res = I.round(N=1)
        # Note: np.round rounds .5 to nearest even number
        expected = Interval([-1.6, 1.1], [1.2, 2.9])
        assert res.isequal(expected)
        
    def test_round_empty(self):
        I = Interval.empty(2)
        res = I.round()
        assert res.is_empty()
        assert res.dim() == 2
        
    def test_round_invalid_N(self):
        I = Interval(1,2)
        with pytest.raises(ValueError):
            I.round(N=-1)
        with pytest.raises(ValueError):
            I.round(N=1.5)

    def test_round_builtin(self):
        I = Interval([-1.55, 1.11], [1.22, 2.88])
        res = round(I, 1)
        # Note: np.round rounds .5 to nearest even number
        expected = Interval([-1.6, 1.1], [1.2, 2.9])
        assert res.isequal(expected) 