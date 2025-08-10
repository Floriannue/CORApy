"""
priv_andEllipsoidIA - Inner approximation of intersection of multiple ellipsoids (at most one degenerate)
CVXPY implementation based on MATLAB description.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_andEllipsoidIA(E_list: List[Ellipsoid]) -> Ellipsoid:
    from cora_python.contSet.interval import Interval
    # At most one degenerate
    if sum([not Ei.isFullDim() for Ei in E_list]) > 1:
        raise CORAerror('CORA:degenerateSet', 'At most one ellipsoid can be degenerate!')

    # Trivial equality shortcut
    if all(E_list[0].isequal(Ei, max(E_list[0].TOL, Ei.TOL)) for Ei in E_list[1:]):
        return E_list[0].copy()

    # Move degenerate to end
    E_nd = [Ei for Ei in E_list if Ei.isFullDim()]
    E_deg = [Ei for Ei in E_list if not Ei.isFullDim()]
    E_cell = E_nd + E_deg

    n = E_cell[0].dim()
    T = np.eye(n)
    x_rem = np.zeros((0, 1))

    if len(E_deg) == 1:
        E_last = E_cell[-1]
        T, S, Vh = np.linalg.svd(E_last.Q)
        nt = E_last.rank()
        # transform
        E_cell_t = []
        for Ei in E_cell:
            Qt = T.T @ Ei.Q @ T
            qt = T.T @ Ei.q
            E_cell_t.append(Ellipsoid(Qt, qt))
        x_rem = E_cell_t[-1].q[nt:]
        E_cell_t[-1] = Ellipsoid(E_cell_t[-1].Q[:nt, :nt], E_cell_t[-1].q[:nt])
        for i in range(len(E_cell_t) - 1):
            E_cell_t[i] = Ellipsoid(E_cell_t[i].Q[:nt, :nt], E_cell_t[i].q[:nt])
        E_cell = E_cell_t
        n = nt

    # 1D case -> exact via interval
    if n == 1:
        Ires = Interval(E_cell[0])
        for Ei in E_cell[1:]:
            Ires = Ires.and_(Interval(Ei), 'exact')
        if Ires.representsa_('emptySet', np.finfo(float).eps):
            return Ellipsoid.empty(E_list[0].dim())
        qt = Ires.center()
        Qt = Ires.rad() ** 2
        Qt_full = np.block([[Qt, np.zeros((n, 0))], [np.zeros((0, 1)), np.zeros((0, 0))]])
        qt_full = np.vstack([qt, x_rem])
        return T @ Ellipsoid(Qt_full, qt_full)

    # CVXPY: maximize det(B)^(1/n) s.t. LMIs
    B = cp.Variable((n, n), symmetric=True)
    d = cp.Variable((n, 1))
    l = [cp.Variable(nonneg=True) for _ in E_cell]

    eps = 1e-9
    constraints = [B >> eps * np.eye(n)]
    I_n = cp.Constant(np.eye(n))
    Z1n = cp.Constant(np.zeros((1, n)))
    Zn1 = cp.Constant(np.zeros((n, 1)))
    for Ei, li in zip(E_cell, l):
        q = cp.Constant(Ei.q)
        Q = cp.Constant(Ei.Q)
        # Build symmetric block matrix
        top = cp.hstack([cp.reshape(1 - li, (1, 1)), Z1n, (d - q).T])
        mid = cp.hstack([Zn1, li * I_n, B])
        bot = cp.hstack([(d - q), B, Q])
        lmi = cp.vstack([top, mid, bot])
        constraints.append(lmi >> 0)

    prob = cp.Problem(cp.Maximize(cp.log_det(B)), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise CORAerror('CORA:solverIssue', f'Inner intersection SDP failed: {prob.status}')

    Bsol = B.value
    dsol = d.value
    Qt = Bsol @ Bsol
    qt = dsol

    if len(E_deg) == 1:
        Nfull = E_list[0].dim()
        Qt_full = np.block([[Qt, np.zeros((n, Nfull - n))], [np.zeros((Nfull - n, n)), np.zeros((Nfull - n, Nfull - n))]])
        qt_full = np.vstack([qt, x_rem])
        return Ellipsoid(T @ Qt_full @ T.T, T @ qt_full)
    else:
        return Ellipsoid(Qt, qt)

