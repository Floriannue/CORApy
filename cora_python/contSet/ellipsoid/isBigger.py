"""
isBigger - checks if an ellipsoid is bigger than another ellipsoid when
   both centers are moved to the origin, i.e.,
   ellipsoid(E2.Q) \subseteq ellipsoid(E1.Q)

Syntax:
   res = isBigger(E1,E2)

Inputs:
   E1 - ellipsoid object
   E2 - ellipsoid object

Outputs:
   res - true/false

Authors:       Victor Gassmann
Written:       10-June-2022
Last update:   20-March-2023 (VG, allow degeneracy)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.sets.contSet.ellipsoid.simdiag import simdiag


def isBigger(E1: Ellipsoid, E2: Ellipsoid) -> bool:
    # check input
    inputArgsCheck([
        [E1, 'att', 'ellipsoid', ['scalar']],
        [E2, 'att', 'ellipsoid', ['scalar']],
    ])

    # check dimensions
    equal_dim_check(E1, E2)

    # set tolerance
    tol = min(E1.TOL, E2.TOL)

    # catch 1D case
    if E1.dim() == 1:
        return True

    # Handle degeneracy cases mirroring MATLAB logic
    if (not E1.isFullDim()) and E2.isFullDim():
        return False
    elif (not E1.isFullDim()) and (not E2.isFullDim()):
        # check for common subspace
        Q_sum = 0.5 * (E1.Q + E2.Q)
        U, S, _ = np.linalg.svd(Q_sum)
        r = np.linalg.matrix_rank(Q_sum)
        if r == E1.dim():
            return False
        # transform and cut to common subspace
        from cora_python.contSet.ellipsoid.project import project
        E1r = project(Ellipsoid(U.T @ E1.Q @ U, U.T @ E1.q), list(range(1, r + 1)))
        E2r = project(Ellipsoid(U.T @ E2.Q @ U, U.T @ E2.q), list(range(1, r + 1)))
        return isBigger(E1r, E2r)

    # simultaneous diagonalization: Tb'*Q1*Tb = I and Tb'*Q2*Tb = D
    _, D = simdiag(E1.Q, E2.Q, tol)
    tmp = np.max(np.diag(D))
    return bool((tmp < 1 + tol) or (abs(tmp - (1 + tol)) <= tol))

