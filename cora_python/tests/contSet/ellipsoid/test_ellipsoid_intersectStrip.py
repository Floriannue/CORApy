import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid


def test_intersectStrip_gollamudi1996_basic():
    C = np.array([[1.0, 0.0]])
    phi = np.array([[0.5]])
    y = np.array([[0.0]])

    E = Ellipsoid(np.array([[1.0, 0.5], [0.5, 1.0]]))

    Eres, sigma_sq = E.intersectStrip(C, phi, y, 0.5, 'Gollamudi1996')
    assert isinstance(Eres, Ellipsoid)
    assert isinstance(sigma_sq, float)


def test_intersectStrip_liu2016_basic():
    C = np.array([[1.0, 0.0]])
    # sys dict mimicking MATLAB struct with required fields
    sys = {
        'bar_V': np.eye(1),
        'V': np.eye(1),
    }
    y = np.array([[0.0]])

    E = Ellipsoid(np.array([[1.0, 0.5], [0.5, 1.0]]))

    Eres, sigma_sq = E.intersectStrip(C, sys, y, 0.5, 'Liu2016')
    assert isinstance(Eres, Ellipsoid)
    assert isinstance(sigma_sq, float)

