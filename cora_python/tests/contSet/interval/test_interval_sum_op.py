import numpy as np
import pytest
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestIntervalSum:
    def test_sum_default(self):
        I = Interval(np.array([[-3, 4], [-2, 1], [-4, 6]]), np.array([[-2, 5], [-1, 2], [-3, 7]]))
        res = I.sum()
        expected = Interval([-9, 11], [-6, 14])
        assert res.isequal(expected)

    def test_sum_dim0(self):
        I = Interval(np.array([[-3, 4], [-2, 1], [-4, 6]]), np.array([[-2, 5], [-1, 2], [-3, 7]]))
        res = I.sum(dim=0)
        expected = Interval([-9, 11], [-6, 14])
        assert res.isequal(expected)

    def test_sum_dim1(self):
        I = Interval(np.array([[-3, 4], [-2, 1], [-4, 6]]), np.array([[-2, 5], [-1, 2], [-3, 7]]))
        res = I.sum(dim=1)
        expected = Interval([1, -1, 2], [3, 1, 4])
        assert res.isequal(expected)

    def test_sum_vector(self):
        I = Interval([-3, -2, -4], [4, 1, 6])
        res = I.sum()
        expected = Interval([-9], [11])
        assert res.isequal(expected)
        
    def test_sum_empty(self):
        I = Interval.empty(2)
        res = I.sum()
        assert res.is_empty()

    def test_sum_default_col_vector(self):
        I = Interval(np.array([[-3], [4]]), np.array([[-2], [5]]))
        res = I.sum()
        expected = Interval([1], [3])
        assert res.isequal(expected)

    def test_sum_default_row_vector(self):
        I = Interval(np.array([[-3, 4]]), np.array([[-2, 5]]))
        res = I.sum()
        expected = Interval([1], [3])
        assert res.isequal(expected)

    def test_sum_invalid_dim(self):
        I = Interval([1, 2], [3, 4])
        with pytest.raises(CORAerror) as e:
            I.sum(dim=2)
        assert "wrongValue" in str(e.value) 