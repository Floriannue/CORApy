import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

class TestEllipsoidVertices:
    def test_empty_case(self):
        n = 2
        E = Ellipsoid.empty(n)
        V = E.vertices_()
        assert isinstance(V, np.ndarray)
        assert V.size == 0 or V.shape[0] == n

    def test_1d_point(self):
        E = Ellipsoid(np.zeros((1, 1)), np.ones((1, 1)))
        V = E.vertices_()
        assert V.shape[1] == 1
        assert np.allclose(V, 1)

    def test_1d_bounded_line(self):
        E = Ellipsoid(np.array([[4]]), np.array([[-1]]))
        V = E.vertices_()
        assert V.shape[1] == 2
        assert np.allclose(V, np.array([[-3, 1]]))

    def test_not_supported_dim(self):
        E = Ellipsoid(np.eye(2), np.zeros((2, 1)))
        with pytest.raises(Exception):
            E.vertices_() 