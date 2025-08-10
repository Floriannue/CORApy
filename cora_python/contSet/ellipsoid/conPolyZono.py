"""
conPolyZono - Converts an ellipsoid to a constrained polynomial zonotope

Syntax:
   cPZ = conPolyZono(E)

Authors:       Niklas Kochdumper
Written:       12-August-2019
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    from cora_python.contSet.conPolyZono import ConPolyZono

def conPolyZono(E: Ellipsoid) -> "ConPolyZono":
    from cora_python.contSet.conPolyZono import ConPolyZono
    # check input arguments
    inputArgsCheck([[E, 'att', 'ellipsoid', ['scalar']]])

    # eigenvalue decomposition of the ellipsoid matrix
    w, V = np.linalg.eigh(E.Q)
    D_sqrt = np.sqrt(np.clip(w, 0, None))

    # dimension and starting point
    n = E.dim()
    c = E.q

    # dependent generator matrix and exponent matrix
    G = V @ np.diag(D_sqrt)
    E_exp = np.vstack([np.eye(n, dtype=int), np.zeros((1, n), dtype=int)])

    # constraints
    A = np.hstack([-0.5 * np.ones((1, 1)), np.ones((1, n))])
    b = np.array([[0.5]])
    EC = np.hstack([np.vstack([np.zeros((n, 1)), np.ones((1, 1))]), np.vstack([2 * np.eye(n), np.zeros((1, n))])])
    # identifiers
    id_vec = np.arange(1, n + 2).reshape(-1, 1)

    # instantiate the constrained polynomial zonotope
    return ConPolyZono(c, G, E_exp, A, b, EC, np.zeros((n, 0)), id_vec)

