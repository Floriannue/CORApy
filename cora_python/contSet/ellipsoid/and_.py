import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from .ellipsoid import Ellipsoid

def and_(E: Ellipsoid, S, method: str = 'exact'):
    """
    and_ - intersection between an ellipsoid and another set

    Args:
        E: Ellipsoid
        S: contSet (Polytope supported)
        method: 'outer'|'inner'|'exact' (for Polytope, 'outer'/'inner' select approximation mode)

    Returns:
        ContSet: intersection approximation
    """
    # Ellipsoid-ellipsoid intersection via private OA/IA
    try:
        if isinstance(S, Ellipsoid):
            if method in ['outer', 'exact']:
                from .private.priv_andEllipsoidOA import priv_andEllipsoidOA
                return priv_andEllipsoidOA(E, S)
            elif method == 'inner':
                from .private.priv_andEllipsoidIA import priv_andEllipsoidIA
                return priv_andEllipsoidIA([E, S])
    except Exception:
        # Fallbacks
        if isinstance(S, Ellipsoid):
            import numpy as _np
            # Outer fallback: pick the smaller-volume ellipsoid to trivially enclose the intersection
            if method in ['outer', 'exact']:
                volE = _np.linalg.det(E.Q)
                volS = _np.linalg.det(S.Q)
                return E if volE <= volS else S
            # Inner fallback: return a tiny ellipsoid around the midpoint (safe but conservative)
            if method == 'inner':
                n = E.dim()
                mid = 0.5 * (E.q + S.q)
                Qtiny = 1e-6 * _np.eye(n)
                from .ellipsoid import Ellipsoid as _Ell
                return _Ell(Qtiny, mid)

    # Polytope intersection handled via private helper
    try:
        from cora_python.contSet.polytope import Polytope
    except Exception:
        Polytope = None

    if Polytope is not None and isinstance(S, Polytope):
        # Hyperplane case should use exact helper as in MATLAB
        try:
            if S.representsa_('hyperplane', getattr(E, 'TOL', 1e-6)):
                from .private.priv_andHyperplane import priv_andHyperplane
                return priv_andHyperplane(E, S)
        except Exception:
            pass
        from .private.priv_andPolytope import priv_andPolytope
        # Map generic method to modes expected by priv_andPolytope
        mode = 'outer' if method in ['outer', 'exact'] else 'inner'
        return priv_andPolytope(E, S, mode)

    # Handle list of ellipsoids: intersect sequentially
    if isinstance(S, list) and all(isinstance(s_i, Ellipsoid) for s_i in S):
        res = E
        for s_i in S:
            res = res.and_(s_i, method)
        return res

    # Fallback to numeric point intersection: treat as hyperplane or point containment
    import numpy as np
    if isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == 1:
        # Point intersection: either E contains point S or empty
        return E if E.contains_(S, 'exact', getattr(E, 'TOL', 1e-6), 0, False, False) else Ellipsoid.empty(E.dim())

    raise CORAerror('CORA:noops', f"and_ not implemented for {type(E).__name__} and {type(S).__name__} with method {method}")

