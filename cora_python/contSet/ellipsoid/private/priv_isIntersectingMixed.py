"""
priv_isIntersectingMixed - checks whether an ellipsoid intersects another set
"""

from __future__ import annotations

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_isIntersectingMixed(E: Ellipsoid, S) -> bool:
    # If degenerate, bloat degenerate dimensions
    if not E.isFullDim():
        T, D, Vh = np.linalg.svd(E.Q)
        r = E.rank()
        n_rem = E.dim() - r
        d = np.diag(np.diag(D)) if D.ndim == 2 else np.diag(D)
        if d.ndim == 2:
            dv = np.diag(d)
        else:
            dv = d
        D_new = np.diag(np.concatenate([dv[:r], 2 * E.TOL * np.max(dv) * np.ones(n_rem)]))
        E = Ellipsoid(T @ D_new @ T.T, E.q)

    # compute interval bounds for (x-q)' Q^{-1} (x-q)
    S_shift = S + (-E.q)
    from cora_python.contSet.interval import Interval
    # quadMap(S, {inv(Q)})
    invQ = np.linalg.pinv(E.Q)
    try:
        quad = S_shift.quadMap([invQ])
    except Exception:
        # If S does not implement quadMap, cannot proceed
        return False
    I = Interval(quad)

    # Check intersection with (0,1)
    return not I.and_(Interval(0, 1), 'exact').representsa_('emptySet', np.finfo(float).eps)

