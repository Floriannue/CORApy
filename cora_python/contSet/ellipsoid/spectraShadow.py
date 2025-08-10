"""
spectraShadow - converts an ellipsoid to a spectrahedral shadow

Syntax:
   SpS = spectraShadow(E)

Authors:       Adrian Kulmburg
Written:       01-August-2023
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.spectraShadow import SpectraShadow

def spectraShadow(E: Ellipsoid) -> "SpectraShadow":
    from cora_python.contSet.spectraShadow import SpectraShadow
    # Separate inverse of shape depending on degeneracy
    if E.isFullDim():
        A0, Ai = _aux_nondegenerate(E)
    else:
        A0, Ai = _aux_degenerate(E)

    # concatenate everything: A = [A0, A1, ..., An]
    A = np.concatenate([A0] + Ai, axis=1)

    # instantiate spectraShadow
    SpS = SpectraShadow(A)

    # additional properties
    SpS.bounded.val = True
    SpS.emptySet.val = E.representsa_('emptySet', 1e-10)
    SpS.fullDim.val = E.isFullDim()
    SpS.center.val = E.center()
    return SpS


def _aux_nondegenerate(E: Ellipsoid):
    n = E.dim()
    L = np.linalg.cholesky(np.linalg.pinv(E.Q))
    # A0 and Ai as in MATLAB comments
    A0 = np.block([[1 - float(E.q.T @ (L.T @ L) @ E.q), np.zeros((1, n))],
                   [np.zeros((n, 1)), np.eye(n)]])
    Ai: List[np.ndarray] = []
    temp = 2 * (E.q.T @ (L.T @ L)).flatten()
    for i in range(n):
        Ai.append(np.block([[temp[i], L[:, i][None, :]],
                            [L[:, i][:, None], np.zeros((n, n))]]))
    return A0, Ai


def _aux_degenerate(E: Ellipsoid):
    n = E.dim()
    U, D, Vt = np.linalg.svd(E.Q)
    # pseudo-inverse sqrt
    U_p, D_p, V_p = np.linalg.svd(np.linalg.pinv(E.Q))
    L = (np.sqrt(D_p)[:, None] * V_p).T

    A0 = np.block([[1 - float(E.q.T @ (L.T @ L) @ E.q), np.zeros((1, n))],
                   [np.zeros((n, 1)), np.eye(n)]])
    Ai: List[np.ndarray] = []
    temp = 2 * (E.q.T @ (L.T @ L)).flatten()
    for i in range(n):
        Ai.append(np.block([[temp[i], L[:, i][None, :]],
                            [L[:, i][:, None], np.zeros((n, n))]]))

    r = int(np.linalg.matrix_rank(E.Q))
    T = np.block([[np.zeros((r, r)), np.zeros((r, n - r))],
                  [np.zeros((n - r, r)), np.eye(n - r)]]) @ U.T
    t = np.block([[np.zeros((r, r))], [np.eye(n - r)]]).T @ E.q
    for i in range(n):
        A0 = np.block([[A0, np.zeros((A0.shape[0], 2))],
                       [np.zeros((2, A0.shape[1])), np.array([[t[i, 0], 0], [0, -t[i, 0]]])]])
        new_blocks = []
        for j in range(n):
            new_blocks.append(np.array([[ -T[i, j], 0],[0, T[i, j]]]))
        # expand Ai[i] blocks
        for k in range(n):
            Ai[k] = np.block([[Ai[k], np.zeros((Ai[k].shape[0], 2))],
                              [np.zeros((2, Ai[k].shape[1])), new_blocks[k]]])
    return A0, Ai

