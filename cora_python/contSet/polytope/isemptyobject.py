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
    This function returns the value of the `_emptySet` property,
    which is computed during the object's construction, mirroring
    the MATLAB implementation.
    
    Args:
        P: polytope object
        
    Returns:
        res: true if polytope is empty, false otherwise
    """
    
    # The emptiness is determined by the lazy-loaded emptySet property.
    return P.emptySet 