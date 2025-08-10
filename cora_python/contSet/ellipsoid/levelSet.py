"""
levelSet - Converts an ellipsoid to a level set

Syntax:
   ls = levelSet(E)

Inputs:
   E - ellipsoid object

Outputs:
   ls - levelSet object

Authors:       Niklas Kochdumper
Written:       09-April-2023
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.levelSet import LevelSet

def levelSet(E: Ellipsoid) -> "LevelSet":
    # check input arguments
    inputArgsCheck([[E, 'att', 'ellipsoid', ['scalar']]])

    # construct level set (sympy-like symbolic form via strings)
    n = E.Q.shape[0]
    # Define variables x1..xn
    vars_ = [f"x{i+1}" for i in range(n)]

    # Build quadratic form (x - q)' inv(Q) (x - q) - 1 <= 0
    Q_inv = np.linalg.pinv(E.Q) if E.Q.size > 0 else np.zeros((0, 0))
    # Create a symbolic string expression for eq
    # eq(x) = (x - q)^T * Q_inv * (x - q) - 1
    # We expand as sum_{i,j} (xi-qi) * Q_inv[i,j] * (xj-qj) - 1
    terms = []
    for i in range(n):
        for j in range(n):
            coef = Q_inv[i, j]
            if abs(coef) > 0:
                ti = f"({vars_[i]}-({float(E.q[i,0])}))"
                tj = f"({vars_[j]}-({float(E.q[j,0])}))"
                terms.append(f"({coef})*{ti}*{tj}")
    eq_str = "+".join(terms) if terms else "0"
    eq_str = f"{eq_str}-1"
    
    from cora_python.contSet.levelSet import LevelSet
    return LevelSet(eq_str, vars_, "<=")

