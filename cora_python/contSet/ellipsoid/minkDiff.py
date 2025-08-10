"""
minkDiff - computes the Minkowski difference of an ellipsoid as a minuend
   and a set as a subtrahend

Syntax:
   E = minkDiff(E,S)
   E = minkDiff(E,S,mode)
   E = minkDiff(E,S,mode,L)

Authors:       Victor Gassmann
Written:       09-March-2021
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def minkDiff(E: Ellipsoid, S, mode: str = None, L: np.ndarray | None = None):
    (mode_val, L_val), _ = setDefaultValues(['outer', np.zeros((E.dim(), 0))], mode, L)
    mode = mode_val
    L = L_val

    # input checks
    inputArgsCheck([
        [E, 'att', 'ellipsoid', ['scalar']],
        [S, 'att', ['contSet', 'numeric', 'cell']],
        [mode, 'str', ['inner', 'outer']],
        [L, 'att', 'numeric'],
    ])

    # check dims
    equal_dim_check(E, S)
    equal_dim_check(E, L)

    # subtrahend is the empty set -> fullspace
    if not isinstance(S, list) and hasattr(S, 'representsa_') and S.representsa_('emptySet', np.finfo(float).eps):
        from cora_python.contSet.fullspace import Fullspace
        return Fullspace(E.dim())

    # numeric column vector
    if isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == 1:
        s = np.sum(S, axis=1, keepdims=True)
        return Ellipsoid(E.Q, E.q - s)

    # list of ellipsoids
    if isinstance(S, list) and all(isinstance(S_i, Ellipsoid) for S_i in S):
        out = Ellipsoid(E)
        from cora_python.contSet.ellipsoid.private.priv_minkDiffEllipsoid import priv_minkDiffEllipsoid
        for S_i in S:
            out = priv_minkDiffEllipsoid(out, S_i, L, mode)
        return out

    # single ellipsoid
    if isinstance(S, Ellipsoid):
        from cora_python.contSet.ellipsoid.private.priv_minkDiffEllipsoid import priv_minkDiffEllipsoid
        return priv_minkDiffEllipsoid(E, S, L, mode)

    raise CORAerror('CORA:noops', E, S)

