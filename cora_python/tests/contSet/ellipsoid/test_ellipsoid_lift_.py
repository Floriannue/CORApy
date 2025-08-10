import numpy as np
import pytest

from cora_python.contSet.ellipsoid import Ellipsoid


def test_lift_same_length_proj_uses_project():
    Q = np.eye(2)
    q = np.array([[0.0], [0.0]])
    E = Ellipsoid(Q, q)
    out = E.lift_(2, [1, 2])
    assert isinstance(out, Ellipsoid)


def test_lift_invalid_raises():
    Q = np.eye(2)
    q = np.array([[0.0], [0.0]])
    E = Ellipsoid(Q, q)
    with pytest.raises(Exception):
        E.lift_(3, [1, 2])

