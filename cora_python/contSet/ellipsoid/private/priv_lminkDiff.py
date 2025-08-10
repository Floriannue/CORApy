"""
priv_lminkDiff - Approximate the Minkowski difference of an ellipsoid and
another ellipsoid or a vector along given directions (MATLAB parity)

Syntax:
   E_cell = priv_lminkDiff(E1,E2,L,mode)

Inputs:
   E1 - ellipsoid object
   E2 - ellipsoid object or numerical vector
   L  - unit vectors in different directions (d x k)
   mode - 'inner' or 'outer'

Outputs:
   E_cell - list of ellipsoids, length equals number of directions
"""

from __future__ import annotations

import numpy as np
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from scipy.linalg import sqrtm
from cora_python.g.functions.helper.sets.contSet.ellipsoid.vecalign import vecalign


def priv_lminkDiff(E1: Ellipsoid, E2, L: np.ndarray, mode: str) -> List[Ellipsoid]:
    # Check bad directions handled by caller; ensure normalization
    norms = np.linalg.norm(L, axis=0, keepdims=True)
    Ln = L / (norms + 1e-16)
    result = []
    for i in range(Ln.shape[1]):
        q = E1.q - (E2.q if isinstance(E2, Ellipsoid) else np.asarray(E2).reshape(-1, 1))
        l = Ln[:, [i]]
        if isinstance(E2, Ellipsoid):
            if mode == 'outer':
                Q1s = sqrtm(E1.Q)
                Q2s = sqrtm(E2.Q)
                Q_ = Q1s - vecalign(Q1s @ l, Q2s @ l) @ Q2s
                Q = Q_.T @ Q_
            else:
                lql1 = float(np.sqrt(l.T @ E1.Q @ l))
                lql2 = float(np.sqrt(l.T @ E2.Q @ l))
                p = lql1 / (lql2 + 1e-16)
                Q = (1 - 1 / p) * E1.Q + (1 - p) * E2.Q
        else:
            Q = E1.Q
        result.append(Ellipsoid(Q, q))
    return result

