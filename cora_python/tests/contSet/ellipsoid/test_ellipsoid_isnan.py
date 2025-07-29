import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.isnan import isnan


def test_ellipsoid_isnan_basic():
    """Test basic isnan functionality for Ellipsoid."""
    Q = np.eye(2)
    E = Ellipsoid(Q)
    assert not isnan(E), "Ellipsoid should not contain NaN values."

def test_ellipsoid_isnan_empty():
    """Test isnan for an empty Ellipsoid (should also be false)."""
    Q_empty = np.array([[]]).reshape(0,0)
    E_empty = Ellipsoid(Q_empty, np.array([]).reshape(0,1))
    assert not isnan(E_empty), "Empty Ellipsoid should not contain NaN values." 