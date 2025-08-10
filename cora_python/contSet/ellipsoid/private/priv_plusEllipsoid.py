"""
priv_plusEllipsoid - Computes inner/outer approximation of Minkowski sum of ellipsoids
"""

from __future__ import annotations

import numpy as np
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_lplus import priv_lplus
from cora_python.contSet.ellipsoid.private.priv_plusEllipsoidOA import priv_plusEllipsoidOA


def priv_plusEllipsoid(E_cell: List[Ellipsoid], L: np.ndarray, mode: str) -> Ellipsoid:
    n = E_cell[0].dim()

    # collapse points and sum their centers
    idx_is_point = [E_i.representsa_('point', np.finfo(float).eps) for E_i in E_cell]
    q_shift = np.zeros((n, 1))
    for i, is_pt in enumerate(idx_is_point):
        if is_pt:
            q_shift += E_cell[i].q
    if all(idx_is_point):
        return Ellipsoid(np.zeros((n, n)), q_shift)
    E_cell = [E_cell[i] for i in range(len(E_cell)) if not idx_is_point[i]]
    E_cell[0].q = E_cell[0].q + q_shift

    if len(E_cell) == 1:
        return E_cell[0]

    # If directions provided
    if L.size > 0:
        L = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-16)
        E_L = priv_lplus(E_cell, L, mode)
        if mode == 'outer':
            from cora_python.contSet.ellipsoid.and_ import and_ as ellipsoid_and
            return ellipsoid_and(E_L[0], E_L[1:], 'outer')
        elif mode == 'inner':
            from cora_python.contSet.ellipsoid.or_ import or_ as ellipsoid_or
            return ellipsoid_or(E_L[0], E_L[1:], 'inner')
        else:
            raise ValueError("mode must be 'inner' or 'outer'")

    # otherwise use Halder (outer) or raise for inner
    if mode == 'outer':
        return priv_plusEllipsoidOA(E_cell)
    elif mode == 'outer:halder':
        return priv_plusEllipsoidOA(E_cell)
    elif mode == 'inner':
        raise NotImplementedError('Inner exact SDP for Minkowski sum not implemented yet')
    else:
        raise ValueError("mode must be 'outer' or 'outer:halder' or 'inner'")

