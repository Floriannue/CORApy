import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.spectraShadow import SpectraShadow


def test_spectraShadow_basic():
    Q = np.array([[3.0, -1.0], [-1.0, 1.0]])
    q = np.array([[1.0], [0.0]])
    E = Ellipsoid(Q, q)
    SpS = E.spectraShadow()
    assert isinstance(SpS, SpectraShadow)
    # structure of A: stored as sparse CSR; check shape consistent with n=2
    assert hasattr(SpS, 'A')
    A = SpS.A
    # A should be k x k*(n+1) block matrix; just check it's 2D and nonempty
    assert hasattr(A, 'shape')
    assert A.shape[0] > 0 and A.shape[1] > 0

