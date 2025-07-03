import numpy as np
import pytest
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestIntervalMinkDiff:
    def test_minkdiff_interval_numeric(self):
        I = Interval([-2, -1], [3, 3])
        S = np.array([-1, 1])
        res = I.minkDiff(S)
        expected = I + (-S)
        assert res.isequal(expected)

    def test_minkdiff_interval_interval(self):
        I1 = Interval([-2, -1], [3, 3])
        I2 = Interval([-1, -1], [1, 1])
        res = I1.minkDiff(I2)
        expected = Interval(I1.inf - I2.inf, I1.sup - I2.sup)
        assert res.isequal(expected)

    def test_minkdiff_other_contset(self):
        I = Interval([-10, -10], [10, 10])
        Z = Zonotope(np.array([1, 1]), np.array([[1, 0.5], [0.5, 1]]))
        
        res = I.minkDiff(Z)

        # manually compute expected result
        inf = I.inf.copy()
        sup = I.sup.copy()
        n = I.dim()

        for i in range(n):
            temp = np.zeros((n,1))
            temp[i] = 1
            sup[i] = sup[i] - Z.supportFunc_(temp, 'upper')[0]
            inf[i] = inf[i] + Z.supportFunc_(-temp, 'upper')[0]

        expected = Interval(inf, sup)

        assert res.isequal(expected)
            
    def test_minkdiff_invalid_interval_diff(self):
        I1 = Interval([0, 0], [1, 1])
        I2 = Interval([2, 2], [4, 4])  # width(I2) > width(I1)
        res = I1.minkDiff(I2)
        assert res.is_empty()

    def test_minkdiff_exact_type_error(self):
        I = Interval([-10, -10], [10, 10])
        Z = Zonotope(np.array([1, 1]), np.array([[1, 0.5], [0.5, 1]]))
        
        with pytest.raises(CORAerror):
            I.minkDiff(Z, type='exact') 