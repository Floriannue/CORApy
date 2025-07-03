import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestIntervalAtanh:
    
    def test_atanh_valid(self):
        i = Interval(np.array([-0.5]), np.array([0.5]))
        res = i.atanh()
        
        expected_inf = np.arctanh(np.array([-0.5]))
        expected_sup = np.arctanh(np.array([0.5]))
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)

    def test_atanh_matrix_valid(self):
        inf = np.array([[-0.9, -0.1], [0.0, 0.2]])
        sup = np.array([[-0.8, 0.1], [0.1, 0.9]])
        i = Interval(inf, sup)
        res = i.atanh()

        expected_inf = np.arctanh(inf)
        expected_sup = np.arctanh(sup)

        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_atanh_empty(self):
        i = Interval.empty()
        res = i.atanh()
        assert res.is_empty()

    def test_atanh_invalid_inf(self):
        i = Interval(np.array([-2.0]), np.array([0.5]))
        with pytest.raises(CORAerror) as e:
            i.atanh()
        assert e.value.identifier == "CORA:outOfDomain"
        
    def test_atanh_invalid_sup(self):
        i = Interval(np.array([-0.5]), np.array([2.0]))
        with pytest.raises(CORAerror) as e:
            i.atanh()
        assert e.value.identifier == "CORA:outOfDomain"

    def test_atanh_invalid_both(self):
        i = Interval(np.array([-2.0]), np.array([2.0]))
        with pytest.raises(CORAerror) as e:
            i.atanh()
        assert e.value.identifier == "CORA:outOfDomain"
        
    def test_atanh_at_bounds(self):
        i = Interval(np.array([-1.0]), np.array([1.0]))
        res = i.atanh()
        assert np.isneginf(res.inf)
        assert np.isposinf(res.sup) 