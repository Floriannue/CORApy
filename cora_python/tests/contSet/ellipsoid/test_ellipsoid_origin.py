import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

class TestEllipsoidOrigin:
    def test_origin_1d(self):
        E = Ellipsoid.origin(1)
        E_true = Ellipsoid(np.zeros((1, 1)), np.zeros((1, 1)))
        assert np.allclose(E.Q, E_true.Q)
        assert np.allclose(E.q, E_true.q)
        # Contains origin
        assert E.contains_(np.zeros((1, 1)))[0]

    def test_origin_2d(self):
        E = Ellipsoid.origin(2)
        E_true = Ellipsoid(np.zeros((2, 2)), np.zeros((2, 1)))
        assert np.allclose(E.Q, E_true.Q)
        assert np.allclose(E.q, E_true.q)
        # Contains origin
        assert E.contains_(np.zeros((2, 1)))[0]

    @pytest.mark.parametrize("bad_n", [0, -1, 0.5, [1, 2], "text"])
    def test_origin_invalid(self, bad_n):
        with pytest.raises(Exception):
            Ellipsoid.origin(bad_n)

    def test_origin_large_dim(self):
        n = 10
        E = Ellipsoid.origin(n)
        assert E.Q.shape == (n, n)
        assert E.q.shape == (n, 1)
        assert np.allclose(E.Q, 0)
        assert np.allclose(E.q, 0)
        assert E.contains_(np.zeros((n, 1)))[0] 