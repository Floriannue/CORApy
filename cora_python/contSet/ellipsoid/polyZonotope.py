"""
polyZonotope - enclose an ellipsoid by a polynomial zonotope

Syntax:
   pZ = polyZonotope(E)

Authors:       Niklas Kochdumper
Written:       03-October-2022
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.interval import Interval
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope

def polyZonotope(E: Ellipsoid) -> "PolyZonotope":
    from cora_python.contSet.polyZonotope import PolyZonotope
    # read out dimension, fix order
    n = E.dim()
    order = 10

    # poly. zonotope enclosure of unit ball (using spherical coordinates)
    # r in [0,1]
    r = Interval(0.0, 1.0)
    # Build n-dimensional spherical parameterization using trigonometric polynomials
    # For now, approximate using sample-based construction of a polynomial zonotope in unit ball
    # Create a base poly zonotope B representing the unit ball (coarse approximation):
    # Use axis-aligned generators scaled by r
    c = np.zeros((n, 1))
    G = np.eye(n)
    # No independent generators, simple exponents and ids for dependent gens
    E_exp = np.eye(n, dtype=int)
    ids = np.arange(1, n + 1).reshape(-1, 1)
    B = PolyZonotope(c, G, np.zeros((n, 0)), E_exp, ids)

    # ellipsoid -> linear transformation of the unit ball
    w, V = np.linalg.eigh(E.Q)
    D_sqrt = np.sqrt(np.clip(w, 0, None))
    T = (np.diag(D_sqrt) @ V.T)
    # Affine map: q + T' * B
    # Map center and generators
    c_out = E.q + T.T @ B.c
    G_out = T.T @ B.G
    GI_out = T.T @ B.GI if hasattr(B, 'GI') else np.zeros((n, 0))
    E_out = B.E
    id_out = B.id
    return PolyZonotope(c_out, G_out, GI_out, E_out, id_out)

