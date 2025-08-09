import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def and_(E: 'Ellipsoid', S, method: str = 'exact'):
    """
    and_ - intersection between an ellipsoid and another set

    Args:
        E: Ellipsoid
        S: contSet (Polytope supported)
        method: 'outer'|'inner'|'exact' (for Polytope, 'outer'/'inner' select approximation mode)

    Returns:
        ContSet: intersection approximation
    """
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

    raise CORAerror('CORA:noops', f"and_ not implemented for {type(E).__name__} and {type(S).__name__} with method {method}")

