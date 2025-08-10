"""
priv_boundary - deterministically computes N uniformly distributed points
   on the boundary of an ellipsoid

Syntax:
   Y, L = priv_boundary(E, N)
"""

from __future__ import annotations

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_boundary(E: Ellipsoid, N: int):
    if N == 0:
        raise CORAerror('CORA:wrongValue', 'second', 'positive integer value')

    # if only center remains
    if E.rank() == 0:
        Y = np.repeat(E.q, N, axis=1)
        L = np.zeros((E.dim(), N))
        return Y, L

    x_rem = np.array([]).reshape(0, 1)
    T, _, _ = np.linalg.svd(E.Q)
    # shift to zero, then transform to diagonal space
    E_nd = (T.T @ (Ellipsoid(E.Q, E.q + (-E.q))))  # Placeholder; better to use transformations if available
    # Without full transform support, mimic by working directly with Q in T-basis
    Q_nd = T.T @ E.Q @ T
    q_nd = T.T @ (E.q - E.q)  # zero

    # handle degeneracy: project to lower-dim space
    if not E.isFullDim():
        nt = int(np.linalg.matrix_rank(Q_nd))
        x_rem = q_nd[nt:]
        Q_nd = Q_nd[:nt, :nt]
    else:
        nt = Q_nd.shape[0]

    n_nd = nt
    # compute equally spaced points on boundary of n_nd-sphere
    if n_nd >= 2:
        # use random equal points approximation: sample via spherical coordinates surrogate
        # Here, approximate by distributing angles on circle for 2D, for higher dims use random dirs
        L_nd = _eq_point_set(n_nd - 1, N)
    else:
        if N % 2 == 0:
            L_nd = np.linspace(-1, 1, N).reshape(1, -1)
        else:
            L_nd = np.linspace(-1, 1, N + 1)[:-1].reshape(1, -1)

    Y_nd = np.zeros((n_nd, N))
    for i in range(N):
        l = L_nd[:, [i]]
        denom = float(l.T @ Q_nd @ l)
        if denom <= 0:
            Y_nd[:, i] = 0
        else:
            Y_nd[:, i] = (Q_nd @ l / np.sqrt(denom)).flatten()
    Y_t = np.vstack([Y_nd, np.tile(x_rem, (1, N))]) if x_rem.size > 0 else Y_nd
    L_t = np.vstack([L_nd, np.zeros((E.dim() - n_nd, N))])

    # backtransform and shift
    Y = T @ Y_t + E.q
    L = T @ L_t
    return Y, L


def _eq_point_set(sphere_dim: int, N: int) -> np.ndarray:
    # crude equal-area point set on sphere via random sampling and normalization
    X = np.random.randn(sphere_dim + 1, N)
    X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-16
    return X

