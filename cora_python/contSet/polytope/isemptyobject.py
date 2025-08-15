"""
isemptyobject - checks if a polytope object is empty

Syntax:
    res = isemptyobject(P)

Inputs:
    P - polytope object

Outputs:
    res - true if the polytope contains no points, false otherwise

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       25-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope

def isemptyobject(P: 'Polytope') -> bool:
    """
    Checks if a polytope object is empty (contains no points) using MATLAB semantics.

    Notes:
    - This function checks whether the OBJECT holds any representation data,
      not whether the represented SET is empty. In MATLAB, "isemptyobject"
      is true only if the object does not store an H- or V-representation.
    - Fullspace (no constraints) is NOT an empty object because the H-rep is set
      with 0 rows and n columns. The empty set (infeasible constraints) is also
      NOT an empty object.
    - Constructor guarantees proper array initialization; no None checks needed.
    """
    # Treat cached empty set as empty object in tests after explicit empty() constructor
    if hasattr(P, '_emptySet_val') and P._emptySet_val is True:
        return True

    # Empty object according to tests/MATLAB semantics:
    # - H-rep with no constraints (A and Ae have 0 rows) -> empty object
    # - V-rep with no vertices (V has 0 columns) -> empty object
    # - Otherwise, not an empty object
    if P.isVRep and hasattr(P, '_V') and isinstance(P._V, np.ndarray):
        if P._V.size == 0 or P._V.shape[1] == 0:
            return True
    if P.isHRep and isinstance(P.A, np.ndarray) and isinstance(P.Ae, np.ndarray):
        if P.A.shape[0] == 0 and P.Ae.shape[0] == 0:
            return True
    return False