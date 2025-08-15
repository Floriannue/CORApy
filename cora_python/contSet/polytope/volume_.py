"""
volume_ - Volume of a polytope (MATLAB parity where possible)

Uses vertices_() to compute convex hull and volume via scipy.spatial.ConvexHull.
Returns 0 for empty or lower-dimensional sets (degenerate), np.inf for unbounded.
No broad try/except; handles degeneracy explicitly via SVD.
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def volume_(P: 'Polytope') -> float:
    from scipy.spatial import ConvexHull

    # Empty
    if P.representsa_('emptySet', 1e-12):
        return 0.0

    # Unbounded
    if not P.isBounded():
        return np.inf

    # Get vertices in ambient space
    V = P.vertices_()  # (n x m)
    n = P.dim()
    if V.size == 0 or V.shape[1] < n + 1:
        # Not enough vertices to form positive nD volume
        return 0.0

    # Remove translation and analyze degeneracy
    c = np.mean(V, axis=1, keepdims=True)
    Vc = V - c
    U, s, _ = np.linalg.svd(Vc, full_matrices=False)
    tol = 1e-12
    r = int(np.sum(s > tol))
    if r < n:
        # Lower-dimensional hull
        return 0.0

    # Project to orthonormal basis Ur to compute volume robustly
    Ur = U[:, :n]
    Vp = (Ur.T @ Vc).T  # (m x n)
    hull = ConvexHull(Vp)
    return float(hull.volume)


