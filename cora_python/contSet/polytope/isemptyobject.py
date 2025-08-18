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

    Based on MATLAB test cases:
    - No constraints (fullspace): isemptyobject(P) returns true
    - No vertices: isemptyobject(P) returns true  
    - Any constraints or vertices: isemptyobject(P) returns false
    """
    # MATLAB code:
    # res_H = ~P.isHRep.val || (isempty(P.b_.val) && isempty(P.be_.val));
    # res_V = ~P.isVRep.val || (isempty(P.V_.val));
    # res = res_H && res_V;
    
    # Check H-representation: no inequality or equality constraints
    res_H = not P.isHRep or (P.b.size == 0 and P.be.size == 0)
    
    # Check V-representation: no vertices
    res_V = not P.isVRep or (P.V.size == 0)
    
    # Combine information: both must be true for empty object
    return res_H and res_V