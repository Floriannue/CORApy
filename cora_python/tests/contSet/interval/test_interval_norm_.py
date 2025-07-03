import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalNorm:
    def test_norm_2_vector(self):
        I = Interval([-2, 1], [3, 2])
        res = I.norm_(p=2)
        expected = np.linalg.norm([3, 2])
        assert np.isclose(res, expected)

    def test_norm_2_matrix(self):
        I = Interval([[-2, 1], [-4, 0]], [[3, 2], [-3, 5]])
        res = I.norm_(p=2)
        m = np.maximum(np.abs(I.inf), np.abs(I.sup))
        expected = np.linalg.norm(m)
        assert np.isclose(res, expected)

    def test_norm_other_error(self):
        I = Interval([1], [2])
        with pytest.raises(NotImplementedError):
            I.norm_(p=1)
            
    def test_norm_empty(self):
        I = Interval.empty()
        res = I.norm_(p=2)
        assert res == -np.inf 