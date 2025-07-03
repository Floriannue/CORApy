import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval

class TestIntervalAsinh:
    
    def test_asinh_basic(self):
        i = Interval(np.array([-1]), np.array([1]))
        res = i.asinh()
        
        expected_inf = np.arcsinh(np.array([-1]))
        expected_sup = np.arcsinh(np.array([1]))
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)

    def test_asinh_matrix(self):
        inf = np.array([[0, 2], [3, 4]])
        sup = np.array([[1, 3], [5, 5]])
        i = Interval(inf, sup)
        res = i.asinh()

        expected_inf = np.arcsinh(inf)
        expected_sup = np.arcsinh(sup)

        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_asinh_empty(self):
        i = Interval.empty()
        res = i.asinh()
        assert res.is_empty() 