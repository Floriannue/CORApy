"""
priv_plusEllipsoidOA - computes the smallest-volume outer-approximation of the Minkowski sum of ellipsoids
CVXPY implementation adapted from MATLAB formulation.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_plusEllipsoidOA(E_cell: List[Ellipsoid]) -> Ellipsoid:
    N = len(E_cell)
    if N == 1:
        return E_cell[0]

    n = E_cell[0].dim()

    # Bloat degenerate ellipsoids slightly to avoid singular inverses
    max_tol = max(Ei.TOL for Ei in E_cell)
    for i in range(N):
        if not E_cell[i].isFullDim():
            U, s, Vh = np.linalg.svd(E_cell[i].Q)
            nt = E_cell[i].rank()
            s = np.array(s)
            if nt < n:
                fill = 2 * max(s[0] * max_tol, max_tol)
                s[nt:] = fill
            Q_i = U @ np.diag(s) @ U.T
            E_cell[i] = Ellipsoid(0.5 * (Q_i + Q_i.T), E_cell[i].q)

    # Build E0 and per-ellipsoid matrices At_i, bt_i, c_i
    E_blocks = []
    E0 = np.zeros((n, n * N))
    At_list = []
    bt_list = []
    c_list = []
    for i in range(N):
        E_i = np.zeros((n, n * N))
        E_i[:, i * n : (i + 1) * n] = np.eye(n)
        E_blocks.append(E_i)
        E0 = E0 + E_i
        Qinv = np.linalg.pinv(E_cell[i].Q)
        b_i = -Qinv @ E_cell[i].q
        At_list.append(E_i.T @ Qinv @ E_i)
        bt_list.append(E_i.T @ b_i)
        c_list.append(float(E_cell[i].q.T @ Qinv @ E_cell[i].q - 1))

    # Variables: symmetric B (n x n), vector b (n x 1), l (N)
    B = cp.Variable((n, n), symmetric=True)
    b = cp.Variable((n, 1))
    l = cp.Variable(N, nonneg=True)

    constraints = []
    # PSD block constraint
    top_left = E0.T @ B @ E0               # (nN x nN)
    top_right = E0.T @ b                   # (nN x 1)
    bottom_left = (E0.T @ b).T             # (1 x nN)
    bottom_right = cp.Constant([[-1]])     # (1 x 1)
    C_psd = cp.bmat([[top_left, top_right], [bottom_left, bottom_right]])
    constraints.append(C_psd >> 0)

    # For each ellipsoid i: B - l_i * At_i <= 0 and b - l_i * bt_i = 0 and 1 - l_i * c_i >= 0
    for i in range(N):
        constraints.append(B - l[i] * At_list[i] << 0)
        constraints.append(E0.T @ b - l[i] * bt_list[i] == 0)
        constraints.append(1 - l[i] * c_list[i] >= 0)

    # Objective: minimize -logdet(B^{-1}) == maximize logdet(B)
    prob = cp.Problem(cp.Maximize(cp.log_det(B)), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise CORAerror('CORA:solverIssue', f'Minkowski sum OA SDP failed: {prob.status}')

    Bsol = B.value
    bsol = b.value
    Q = np.linalg.pinv(Bsol)
    q = -np.linalg.solve(Bsol, bsol)
    return Ellipsoid(Q, q)

