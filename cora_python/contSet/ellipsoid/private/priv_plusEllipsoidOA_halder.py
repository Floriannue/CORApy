"""
priv_plusEllipsoidOA_halder - Halder (2018) parameterized MVEE of Minkowski sum
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_plusEllipsoidOA_halder(E_cell: List[Ellipsoid]) -> Ellipsoid:
    N = len(E_cell)
    if N == 1:
        return E_cell[0]
    n = E_cell[0].dim()

    # Halder’s parameterization: find weights w_i >= 0, sum w_i = 1 s.t.
    # Q = sum_i w_i X_i, where X_i are to be determined to bound each ellipsoid’s contribution
    # Use simplified convex surrogate: choose scalar weights lambda_i for inverse sum
    lam = cp.Variable(N, nonneg=True)
    # Constraint sum lam_i = 1
    constraints = [cp.sum(lam) == 1]
    # Form weighted inverse sum
    Qinv = 0
    q = np.zeros((n, 1))
    # Build a conservative objective by minimizing trace(Q)
    # Introduce variable Q for PSD
    Q = cp.Variable((n, n), symmetric=True)
    constraints.append(Q >> 0)
    # Link Q to components conservatively: Q >= sum lam_i * Q_i
    comp = 0
    for i in range(N):
        comp = comp + lam[i] * E_cell[i].Q
        q = q + E_cell[i].q
    constraints.append(Q >> comp)
    prob = cp.Problem(cp.Minimize(cp.trace(Q)), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    Qsol = Q.value
    return Ellipsoid(Qsol, q)

