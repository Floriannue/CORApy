import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid


def test_plot3D_runs():
    Q = np.diag([3.0, 2.0, 1.0])
    q = np.zeros((3,1))
    E = Ellipsoid(Q, q)
    # ensure function runs without raising exception (backend-independent)
    # Use a non-interactive backend if needed
    import matplotlib
    matplotlib.use('Agg')
    E.plot3D([0,1,2])

