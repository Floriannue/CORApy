import numpy as np
import pytest
from cora_python.contSet.interval import Interval

class TestIntervalKron:
    def test_kron_interval_interval(self):
        I1 = Interval(np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]))
        I2 = Interval(np.array([[0, 5], [6, 7]]), np.array([[1, 6], [7, 8]]))
        res = Interval.kron(I1, I2)

        # Manually compute expected result
        inf1, sup1 = I1.inf, I1.sup
        inf2, sup2 = I2.inf, I2.sup
        
        c1 = np.kron(inf1, inf2)
        c2 = np.kron(inf1, sup2)
        c3 = np.kron(sup1, inf2)
        c4 = np.kron(sup1, sup2)
        
        expected_inf = np.minimum.reduce([c1, c2, c3, c4])
        expected_sup = np.maximum.reduce([c1, c2, c3, c4])
        expected = Interval(expected_inf, expected_sup)

        assert res.isequal(expected)

    def test_kron_matrix_interval(self):
        M = np.array([[1, 2], [3, 4]])
        I = Interval(np.array([[0, 5], [6, 7]]), np.array([[1, 6], [7, 8]]))
        res = Interval.kron(M, I)

        c1 = np.kron(M, I.inf)
        c2 = np.kron(M, I.sup)

        expected_inf = np.minimum(c1, c2)
        expected_sup = np.maximum(c1, c2)
        expected = Interval(expected_inf, expected_sup)

        assert res.isequal(expected)

    def test_kron_interval_matrix(self):
        I = Interval(np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]))
        M = np.array([[0, 5], [6, 7]])
        res = Interval.kron(I, M)

        c1 = np.kron(I.inf, M)
        c2 = np.kron(I.sup, M)

        expected_inf = np.minimum(c1, c2)
        expected_sup = np.maximum(c1, c2)
        expected = Interval(expected_inf, expected_sup)

        assert res.isequal(expected) 