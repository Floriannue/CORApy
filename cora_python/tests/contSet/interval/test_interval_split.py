import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalSplit:
    def test_split_1d(self):
        I = Interval([-1], [1])
        res = I.split(0)
        
        assert len(res) == 2
        assert res[0].isequal(Interval([-1], [0]))
        assert res[1].isequal(Interval([0], [1]))

    def test_split_2d_dim0(self):
        I = Interval([-1, -1], [1, 1])
        res = I.split(0)
        
        assert len(res) == 2
        assert res[0].isequal(Interval([-1, -1], [0, 1]))
        assert res[1].isequal(Interval([0, -1], [1, 1]))

    def test_split_2d_dim1(self):
        I = Interval([-1, -1], [1, 1])
        res = I.split(1)
        
        assert len(res) == 2
        assert res[0].isequal(Interval([-1, -1], [1, 0]))
        assert res[1].isequal(Interval([-1, 0], [1, 1]))

    def test_split_matrix_error(self):
        I = Interval(np.zeros((2,2)), np.ones((2,2)))
        with pytest.raises(ValueError):
            I.split(0)
            
    def test_split_invalid_dim(self):
        I = Interval([-1], [1])
        with pytest.raises(ValueError):
            I.split(1)
        with pytest.raises(ValueError):
            I.split(-1) 