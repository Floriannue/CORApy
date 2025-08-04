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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope

def isemptyobject(P: 'Polytope') -> bool:
    """
    Checks if a polytope object is empty (contains no points).
    This mirrors the MATLAB implementation exactly.
    
    Args:
        P: polytope object
        
    Returns:
        res: true if polytope is empty, false otherwise
    """
    
    # MATLAB: res_H = ~P.isHRep.val || (isempty(P.b_.val) && isempty(P.be_.val));
    # no inequality or equality constraints
    res_H = not P.isHRep or (P.b.size == 0 and P.be.size == 0)
    
    # MATLAB: res_V = ~P.isVRep.val || (isempty(P.V_.val));
    # no vertices  
    res_V = not P.isVRep or (P.V.size == 0)
    
    # MATLAB: res = res_H && res_V;
    # combine information
    res = res_H and res_V
    
    return res 