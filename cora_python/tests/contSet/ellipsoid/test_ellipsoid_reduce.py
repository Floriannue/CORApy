import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.reduce import reduce

def test_reduce_empty():
    """Test reduce functionality with an empty ellipsoid."""
    E = Ellipsoid.empty(2)
    E_ = reduce(E)
    assert E is E_

def test_reduce_basic():
    """Test reduce functionality with a basic ellipsoid."""
    E = Ellipsoid(np.array([[2, 0], [0, 1]]), np.array([[1], [-1]]))
    E_ = reduce(E)
    assert E is E_