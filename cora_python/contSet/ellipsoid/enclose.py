"""
enclose - encloses an ellipsoid and its affine transformation

Syntax:
   E = enclose(E,E2)
   E = enclose(E,M,Eplus)
"""

from __future__ import annotations

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check


def enclose(E: Ellipsoid, *args):
    if len(args) == 1:
        E2 = args[0]
        inputArgsCheck([[E, 'att', 'ellipsoid', ['scalar']],
                        [E2, 'att', 'ellipsoid', ['scalar']]])
        equal_dim_check(E, E2)
    elif len(args) == 2:
        M, Eplus = args
        inputArgsCheck([[E, 'att', 'ellipsoid', ['scalar']],
                        [Eplus, 'att', 'ellipsoid', ['scalar']],
                        [M, 'att', 'numeric', lambda M: M.shape == (E.dim(), Eplus.dim())]])
        E2 = M @ Eplus
    else:
        raise ValueError('enclose expects 1 or 2 additional arguments')

    return E.convHull(E2)

