import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def priv_isMinVRep(P: 'Polytope') -> bool:
    """
    Checks if the polytope is represented by its minimal vertex representation.

    Args:
        P: Polytope object.

    Returns:
        bool: True if the vertex representation is minimal, False otherwise.
    """
    n = P.dim()

    if P.isVRep:
        if P.V.size == 0:
            # Empty set has a minimal V-representation (no vertices)
            return True
        if n == 1:
            # 1D polytopes always have a minimal V-representation (min/max or single point)
            return True
        else:
            # nD V-representation: minimal if num_vertices <= n+1 and affinely independent
            # or if it's a point (num_vertices == 1).
            # MATLAB's aux_computeHiddenProperties checks size(V,2) <= 1 or n == 1.
            # For nD, it simply means 1 vertex (a point) is minimal.
            # A full check would involve checking affine independence of vertices.
            # For now, follow the direct MATLAB check which is `size(V,2) <= 1` or `n == 1`.
            return P.V.shape[1] <= 1 # True if 0 or 1 vertex
    else:
        # Not a V-representation, so it cannot be a minimal V-representation
        return False 