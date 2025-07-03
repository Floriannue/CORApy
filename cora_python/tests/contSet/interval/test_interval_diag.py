import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalDiag:
    def test_diag_from_vector(self):
        I = Interval([1, -1], [2, 0])
        res = I.diag()
        expected = Interval([[1, 0], [0, -1]], [[2, 0], [0, 0]])
        assert res.isequal(expected)

    def test_diag_from_matrix(self):
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        res = I.diag()
        expected = Interval([1, 4], [2, 5])
        assert res.isequal(expected)

    def test_diag_from_matrix_k1(self):
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        res = I.diag(k=1)
        expected = Interval([2], [3])
        assert res.isequal(expected)
        
    def test_diag_empty(self):
        I = Interval.empty(2)
        res = I.diag()
        assert res.is_empty()

    def test_diag_nd_error(self):
        inf = np.ones((2,2,2))
        sup = 2*np.ones((2,2,2))
        I = Interval(inf, sup)
        with pytest.raises(ValueError):
            I.diag() 