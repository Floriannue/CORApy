"""
hausdorffDist - Hausdorff distance between a polytope and a set/point (MATLAB parity)

Cases handled (per MATLAB doc):
- Identical sets: 0
- bounded + bounded: finite > 0 (or 0 if identical)
- empty + bounded: Inf
- unbounded + bounded: Inf
- unbounded + unbounded (non-identical): Inf

For bounded polytopes, computes Hausdorff distance using vertices:
d_H(P,Q) = max( max_{v in V(Q)} dist(P,v), max_{u in V(P)} dist(Q,u) )
"""

import numpy as np
from typing import TYPE_CHECKING, Union

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def hausdorffDist(P: 'Polytope', S: Union['Polytope', np.ndarray]) -> float:
    from .polytope import Polytope
    from .distance import distance

    tol = 1e-10

    # Dimension check when S is polytope
    if isinstance(S, Polytope) and P.dim() != S.dim():
        raise CORAerror('CORA:wrongDimension', 'Hausdorff distance requires equal dimensions')

    # Special cases for numeric S (point cloud)
    if isinstance(S, np.ndarray):
        # Unbounded P -> Inf
        if not P.isBounded():
            return float('inf')
        # Empty P
        if P.representsa_('emptySet', tol):
            return 0.0 if S.size == 0 else float('inf')
        # Bounded P vs point cloud: use max distance of points to P
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        dists = distance(P, S)
        return float(np.max(dists)) if isinstance(dists, np.ndarray) else float(dists)

    # Polytope vs Polytope cases
    Q: Polytope = S

    # Empty checks first
    P_empty = P.representsa_('emptySet', tol)
    Q_empty = Q.representsa_('emptySet', tol)
    if P_empty and Q_empty:
        return 0.0
    if P_empty and not Q_empty:
        return float('inf')
    if Q_empty and not P_empty:
        return float('inf')

    # Identical after handling emptiness
    if P.isequal(Q):
        return 0.0

    P_bounded = P.isBounded()
    Q_bounded = Q.isBounded()
    if P_bounded and Q_bounded:
        # Compute via vertices
        Vp = P.V if P.isVRep and getattr(P, '_V', None) is not None and P._V.size > 0 else P.vertices_()
        Vq = Q.V if Q.isVRep and getattr(Q, '_V', None) is not None and Q._V.size > 0 else Q.vertices_()
        if Vp.size == 0 or Vq.size == 0:
            return 0.0
        d1 = distance(P, Vq)
        d2 = distance(Q, Vp)
        d1_max = float(np.max(d1)) if isinstance(d1, np.ndarray) else float(d1)
        d2_max = float(np.max(d2)) if isinstance(d2, np.ndarray) else float(d2)
        return max(d1_max, d2_max)

    # If one bounded and one unbounded (non-identical), Inf
    if (P_bounded and not Q_bounded) or (Q_bounded and not P_bounded):
        return float('inf')

    # Both unbounded and non-identical -> Inf per MATLAB doc
    return float('inf')


