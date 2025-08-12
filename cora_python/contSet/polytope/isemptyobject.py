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
    - Fullspace (no constraints) is NOT empty.
    - Constructor guarantees proper array initialization; no None checks needed.
    - Avoid triggering conversions: use internal storages where possible.
    """
    # If cached from constructor (e.g., Polytope.empty), respect it
    if hasattr(P, '_emptySet_val') and P._emptySet_val is not None:
        return bool(P._emptySet_val)

    # MATLAB semantics: object is empty if both H-rep and V-rep storages are empty
    # H-rep considered empty if both b and be are empty
    res_H = (not P.isHRep) or (P.b.size == 0 and P.be.size == 0)
    # V-rep considered empty if V is empty
    # Use internal storage to avoid triggering vertices_()
    res_V = (not P.isVRep) or (P._V.size == 0)
    return bool(res_H and res_V)