"""
priv_andEllipsoidOA - Computes an outer-approximation of the intersection
of two ellipsoids using the MATLAB-aligned parameterization (no SDP).
"""

from __future__ import annotations

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .priv_compIntersectionParam import priv_compIntersectionParam
from .priv_rootfnc import priv_rootfnc


def priv_andEllipsoidOA(E1: Ellipsoid, E2: Ellipsoid) -> Ellipsoid:
    # At least one must be full-dimensional
    if not E1.isFullDim() and not E2.isFullDim():
        raise CORAerror('CORA:degenerateSet', 'At least one ellipsoid has to be full-dimensional!')

    # If exactly equal
    if E1.isequal(E2, max(E1.TOL, E2.TOL)):
        return E1.copy()

    # If not intersecting -> empty
    if not E1.isIntersecting_(E2, 'exact', 1e-8):
        return Ellipsoid.empty(E1.dim())

    n = E1.dim()
    T = np.eye(n)
    x2_rem = np.zeros((0, 1))
    E1p, E2p = E1, E2

    # handle degeneracy: ensure at most E2 is degenerate, project to its non-degenerate subspace
    if not E1.isFullDim() and E2.isFullDim():
        E1p, E2p = E2, E1  # swap so that E2p might be degenerate

    if not E2p.isFullDim():
        nt = E2p.rank()
        U, S, Vt = np.linalg.svd(E2p.Q)
        T = U
        # transform
        def tr(Q, q):
            return T.T @ Q @ T, T.T @ q
        Q2t, q2t = tr(E2p.Q, E2p.q)
        Q1t, q1t = tr(E1p.Q, E1p.q)
        x2_rem = q2t[nt:]
        E2p = Ellipsoid(Q2t[:nt, :nt], q2t[:nt])
        E1p = Ellipsoid(Q1t[:nt, :nt], q1t[:nt])

    # parameterization in reduced space
    W1 = np.linalg.inv(E1p.Q)
    W2 = np.linalg.inv(E2p.Q)
    q1 = E1p.q
    q2 = E2p.q

    p = priv_compIntersectionParam(W1, q1, W2, q2)
    _, Qt, qt = priv_rootfnc(p, W1, q1, W2, q2)

    # check if Qt describes an ellipsoid
    vals = np.linalg.eigvalsh(Qt)
    if np.any(vals < 0):
        return Ellipsoid.empty(E1.dim())

    # backtransform
    if x2_rem.size > 0:
        n_nd = E2p.dim()
        Qt_full = np.block([[Qt, np.zeros((n_nd, n - n_nd))], [np.zeros((n - n_nd, n_nd)), np.zeros((n - n_nd, n - n_nd))]])
        qt_full = np.vstack([qt, x2_rem])
        return Ellipsoid(T @ Qt_full @ T.T, T @ qt_full)
    else:
        return Ellipsoid(Qt, qt)

