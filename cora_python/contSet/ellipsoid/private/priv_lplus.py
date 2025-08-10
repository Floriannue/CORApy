"""
priv_lplus - Computes the Minkowski sum of a list of ellipsoids such that
   the resulting over-approximation is tight in given directions

Syntax:
   E_cell = priv_lplus(E_cell,L,mode)
"""

from __future__ import annotations

import numpy as np
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def priv_lplus(E_cell: List[Ellipsoid], L: np.ndarray, mode: str) -> List[Ellipsoid]:
    # single ellipsoid fast path
    if len(E_cell) == 1:
        return E_cell

    out: List[Ellipsoid] = [None] * L.shape[1]
    for i in range(L.shape[1] - 1, -1, -1):
        out[i] = _aux_lplus_single(E_cell, L[:, i : i + 1], mode)
    return out


def _aux_lplus_single(E_cell: List[Ellipsoid], l: np.ndarray, mode: str) -> Ellipsoid:
    n = E_cell[0].dim()
    l = l.reshape(-1, 1)

    if mode == 'outer':
        q = np.zeros((n, 1))
        c = 0.0
        Q_sum = np.zeros((n, n))
        for E_i in E_cell:
            q += E_i.q
            si = float(np.sqrt(l.T @ E_i.Q @ l))
            c += si
            if not withinTol(si, 0, E_i.TOL):
                Q_sum += E_i.Q / si
        Q = c * Q_sum
        return Ellipsoid(Q, q)
    else:  # 'inner'
        x = _sqrtm(E_cell[0].Q) @ l
        q = np.zeros((n, 1))
        Q_acc = np.zeros((n, n))
        for E_i in E_cell:
            q += E_i.q
            Qs = _sqrtm(E_i.Q)
            Q_acc += _vecalign(x, Qs @ l) @ Qs
        Q = Q_acc.T @ Q_acc
        return Ellipsoid(Q, q)


def _sqrtm(Q: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(Q)
    w_clipped = np.clip(w, 0, None)
    return (V * np.sqrt(w_clipped)) @ V.T


def _vecalign(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a) + 1e-16)
    bn = b / (np.linalg.norm(b) + 1e-16)
    return an @ bn.T

