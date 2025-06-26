import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid

def test_volume_():
    # 2D Ellipsoid
    E2 = Ellipsoid(np.diag([1, 4]), np.array([[1], [-1]]))
    # Expected volume: pi * a * b = pi * sqrt(1) * sqrt(4) = 2*pi
    expected_vol2 = 2 * np.pi
    assert np.isclose(E2.volume_(), expected_vol2)

    # 3D Ellipsoid
    E3 = Ellipsoid(np.diag([1, 4, 9]))
    # Expected volume: (4/3) * pi * a * b * c = (4/3) * pi * 1 * 2 * 3 = 8*pi
    expected_vol3 = 8 * np.pi
    assert np.isclose(E3.volume_(), expected_vol3)

    # Empty Ellipsoid
    E_empty = Ellipsoid.empty()
    assert E_empty.volume_() == 0.0 