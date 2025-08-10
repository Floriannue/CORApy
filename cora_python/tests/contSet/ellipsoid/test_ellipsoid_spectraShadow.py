import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.spectraShadow import SpectraShadow


def test_spectraShadow_basic():
    Q = np.array([[3.0, -1.0], [-1.0, 1.0]])
    q = np.array([[1.0], [0.0]])
    E = Ellipsoid(Q, q)
    SpS = E.spectraShadow()
    assert isinstance(SpS, SpectraShadow)

