"""
compact_ - Remove redundant constraints/vertices (MATLAB parity)

For H-rep: normalize and call priv_compact_all to remove redundancies.
For V-rep: compute convex hull extreme points and store as minimal V-rep.
"""

import numpy as np
from typing import TYPE_CHECKING

from .private.priv_normalizeConstraints import priv_normalizeConstraints
from .private.priv_compact_all import priv_compact_all

if TYPE_CHECKING:
    from .polytope import Polytope


def compact_(P: 'Polytope', method: str = 'all', tol: float = 1e-12) -> 'Polytope':
    # If H-representation available, normalize and compact
    if P.isHRep:
        A, b, Ae, be = priv_normalizeConstraints(P.A, P.b, P.Ae, P.be, 'A')
        # Only 'all' method supported for H-rep compaction via priv_compact_all
        if method != 'all':
            raise ValueError(f"Method '{method}' not supported for H-representation compaction in Polytope.compact_.")
        
        A, b, Ae, be, _, _ = priv_compact_all(A, b, Ae, be, P.dim(), tol)
        P._A, P._b, P._Ae, P._be = A, b, Ae, be
        return P

    # If V-representation, prune to convex hull extreme vertices
    if P.isVRep:
        # Only 'all' method supported for V-rep compaction (convex hull)
        if method != 'all':
            raise ValueError(f"Method '{method}' not supported for V-representation compaction in Polytope.compact_.")

        from scipy.spatial import ConvexHull
        V = P.V
        if V.size == 0 or V.shape[1] <= 1:
            return P
        # Deterministic degeneracy handling via SVD projection
        c = np.mean(V, axis=1, keepdims=True)
        Vc = V - c
        U, s, _ = np.linalg.svd(Vc, full_matrices=False)
        tol = 1e-12
        r = int(np.sum(s > tol))
        if r == 0:
            P._V = c
            return P
        if r == 1:
            u = U[:, [0]]
            coords = (u.T @ Vc).flatten()
            i_min = int(np.argmin(coords)); i_max = int(np.argmax(coords))
            P._V = np.unique(np.round(V[:, [i_min, i_max]].T, 12), axis=0).T
            return P
        Ur = U[:, :r]
        Vp = (Ur.T @ Vc).T
        if Vp.shape[0] <= r:
            return P
        hull = ConvexHull(Vp)
        idx = np.unique(hull.vertices)
        P._V = np.unique(np.round(V[:, idx].T, 12), axis=0).T
        return P

    return P


