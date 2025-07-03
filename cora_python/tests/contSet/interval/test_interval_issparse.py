import numpy as np
import pytest
from scipy.sparse import csr_matrix
from cora_python.contSet.interval import Interval

class TestIntervalIssparse:
    def test_issparse_dense(self):
        I = Interval(np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]]))
        assert not I.issparse()

    def test_issparse_sparse_inf(self):
        inf = csr_matrix([[1, 0], [0, 1]])
        sup = np.array([[2, 0], [0, 2]])
        I = Interval(inf, sup)
        assert I.issparse()

    def test_issparse_sparse_sup(self):
        inf = np.array([[1, 0], [0, 1]])
        sup = csr_matrix([[2, 0], [0, 2]])
        I = Interval(inf, sup)
        assert I.issparse()

    def test_issparse_both_sparse(self):
        inf = csr_matrix([[1, 0], [0, 1]])
        sup = csr_matrix([[2, 0], [0, 2]])
        I = Interval(inf, sup)
        assert I.issparse()

    def test_issparse_empty(self):
        I = Interval.empty()
        assert not I.issparse() 