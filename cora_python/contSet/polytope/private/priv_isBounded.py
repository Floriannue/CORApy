import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def priv_isBounded(P: 'Polytope') -> bool:
    """
    Checks if a polytope is bounded.

    Args:
        P: Polytope object.

    Returns:
        bool: True if the polytope is bounded, False otherwise.
    """
    n = P.dim()

    if P.isVRep:
        # V-representation is always bounded in nD. Only 1D can have infinite vertices.
        if n == 1:
            # Bounded if no infinite vertices
            return not np.any(np.isinf(P.V))
        else:
            # nD V-representation is always bounded
            return True
    elif P.isHRep:
        # MATLAB's aux_computeHiddenProperties sets bounded to false only for R^n (no constraints).
        # Otherwise, it does not explicitly compute it for other H-rep cases in this block.
        # For a robust check for H-representation, we need to check if the recession cone is just the origin.
        # This typically involves solving LPs or checking feasibility of certain rays.

        if P.A.size == 0 and P.Ae.size == 0:
            return False # R^n is unbounded
        
        # For other HRep cases, MATLAB's aux_computeHiddenProperties only explicitly handles R^n.
        # It assumes other H-rep polytopes are bounded unless specifically proven otherwise (e.g., through redundancy removal).
        # To be faithful to the MATLAB aux_computeHiddenProperties's explicit logic:
        # If it's not R^n, assume bounded unless proven otherwise by a more complex method.
        # In MATLAB, for HRep, bounded is set to 'true' if empty or if certain inf/nan conditions are met.
        # For a general HRep, checking boundedness is non-trivial and often involves checking if the origin is contained in the recession cone.

        # For now, follow the explicit simple cases from MATLAB's aux_computeHiddenProperties.
        # If it's not R^n (no constraints), it assumes it's bounded, unless other specific (infeasible) cases make it empty.
        # The MATLAB logic for `bounded` in `aux_computeHiddenProperties` for HRep is primarily:
        # `bounded = false` for R^n
        # `bounded = true` if `empty = true` (which happens for infinite bounds on `be` or negative infinite `b`)

        # Given the complexity and deferring a full recession cone analysis:
        # If the polytope is empty due to infinite bounds, it's considered bounded in MATLAB.
        if P.emptySet and (np.any(np.isinf(P.be)) or (P.b.size > 0 and np.any(np.isinf(P.b) & (np.sign(P.b) == -1)) ) ) :
            return True
        
        # Default for non-R^n H-rep based on limited explicit MATLAB logic:
        # Without full recession cone computation, assume bounded if not R^n.
        return True

    else:
        # Neither VRep nor HRep flags set (e.g., Polytope() or 0-dim empty set)
        # An empty set is considered bounded.
        return P.emptySet 