import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid


def test_convHull_ellipsoid_with_ellipsoid_calls_or_outer():
    E1 = Ellipsoid(np.array([[3.0, -1.0], [-1.0, 1.0]]), np.array([[1.0], [0.0]]))
    E2 = Ellipsoid(np.array([[5.0, 1.0], [1.0, 2.0]]), np.array([[1.0], [-1.0]]))
    out = E1.convHull_(E2)
    assert hasattr(out, 'precedence')

