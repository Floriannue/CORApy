import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid


def test_isBigger_simple():
    Q1 = np.array([[3.0, 0.0], [0.0, 3.0]])
    Q2 = np.array([[2.0, 0.0], [0.0, 2.0]])
    E1 = Ellipsoid(Q1, np.zeros((2, 1)))
    E2 = Ellipsoid(Q2, np.zeros((2, 1)))
    assert E1.isBigger(E2) is True


def test_isBigger_degenerate_case():
    # E1 degenerate along one axis, E2 non-degenerate -> expect False
    Q1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    Q2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    E1 = Ellipsoid(Q1, np.zeros((2, 1)))
    E2 = Ellipsoid(Q2, np.zeros((2, 1)))
    assert E1.isBigger(E2) is False

