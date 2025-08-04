import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.isBounded import isBounded


def test_ellipsoid_isBounded_basic():
    """Test basic isBounded functionality for Ellipsoid."""
    # Create a simple 2x2 identity matrix for Q
    Q = np.eye(2)
    E = Ellipsoid(Q)
    assert isBounded(E), "Ellipsoid should always be bounded."

def test_ellipsoid_isBounded_empty():
    """Test isBounded for an empty Ellipsoid (should still be bounded)."""
    # An empty ellipsoid is represented by a 0x0 shape matrix
    Q_empty = np.array([[]]).reshape(0,0)
    E_empty = Ellipsoid(Q_empty, np.array([]).reshape(0,1))
    assert isBounded(E_empty), "Empty Ellipsoid should also be bounded." 