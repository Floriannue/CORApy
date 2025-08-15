"""
convHull_ - Convex hull of two polytopes (MATLAB parity)

If both inputs are polytopes, return the convex hull as a V-representation
by concatenating vertices and computing the convex hull extreme points.
Deterministic handling of degeneracy without masking errors.
"""

import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from .polytope import Polytope


def convHull_(P1: 'Polytope', P2: 'Polytope') -> 'Polytope':
    from .polytope import Polytope

    n = P1.dim()
    assert n == P2.dim(), "Dimension mismatch in convHull_"

    # Get vertices for both (compute explicitly to avoid stale/cached issues)
    V1 = P1.vertices_()
    V2 = P2.vertices_()

    if V1.size == 0 and V2.size == 0:
        # Both empty
        return Polytope.empty(n)

    if V1.size == 0:
        return Polytope(V2)
    if V2.size == 0:
        return Polytope(V1)

    V = np.hstack([V1, V2])

    # Deduplicate vertices (tolerance-based)
    V = np.unique(np.round(V.T, 12), axis=0).T

    # Handle trivial cases
    if V.shape[1] == 0:
        return Polytope.empty(n)
    if V.shape[1] == 1:
        return Polytope(V)

    # Remove affine degeneracy via projection to subspace, compute hull, and lift back
    c = np.mean(V, axis=1, keepdims=True)
    Vc = V - c
    U, s, _ = np.linalg.svd(Vc, full_matrices=False)
    tol = 1e-12
    r = int(np.sum(s > tol))

    if r == 0:
        # All points coincide
        return Polytope(c)
    if r == 1:
        # 1D hull: take min/max along principal direction
        u = U[:, [0]]
        coords = (u.T @ Vc).flatten()
        i_min = int(np.argmin(coords))
        i_max = int(np.argmax(coords))
        V_ext = V[:, [i_min, i_max]]
        V_ext = np.unique(np.round(V_ext.T, 12), axis=0).T
        return Polytope(V_ext)

    # r >= 2: project and compute convex hull in rD
    Ur = U[:, :r]
    Vp = (Ur.T @ Vc).T  # (m x r)
    # Need at least r+1 points for a non-empty hull
    if Vp.shape[0] <= r:
        return Polytope(V)
    hull = ConvexHull(Vp)
    extreme_idx = np.unique(hull.vertices)
    V_ext = V[:, extreme_idx]
    # Final deduplicate and return
    V_ext = np.unique(np.round(V_ext.T, 12), axis=0).T
    return Polytope(V_ext)


