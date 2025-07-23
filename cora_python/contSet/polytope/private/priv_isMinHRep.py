import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def priv_isMinHRep(P: 'Polytope') -> bool:
    """
    Checks if the polytope is represented by its minimal halfspace representation.

    Args:
        P: Polytope object.

    Returns:
        bool: True if the halfspace representation is minimal, False otherwise.
    """
    n = P.dim()

    if P.isHRep:
        if P.A.size == 0 and P.Ae.size == 0:
            # R^n (no constraints) has a minimal H-representation
            return True
        else:
            # For other H-representations, MATLAB's aux_computeHiddenProperties
            # does not explicitly compute minimal H-representation. It relies
            # on other functions like `minHRep` (a method) to do the actual computation.
            # For now, we return False for these cases, mirroring the limited scope of
            # aux_computeHiddenProperties. A full implementation would involve
            # redundancy removal (e.g., cddlib or scipy.optimize).
            return False
    else:
        # Not an H-representation, so it cannot be a minimal H-representation.
        return False 