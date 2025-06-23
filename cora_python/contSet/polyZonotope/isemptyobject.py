"""
isemptyobject - checks whether a polynomial zonotope contains any information

Syntax:
    res = isemptyobject(pZ)

Inputs:
    pZ - polyZonotope object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       07-June-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def isemptyobject(pZ: 'PolyZonotope') -> bool:
    """
    Checks whether a polynomial zonotope contains any information
    
    Args:
        pZ: polyZonotope object
        
    Returns:
        res: true if the polynomial zonotope is empty, false otherwise
    """
    
    # Check if center is empty
    if not hasattr(pZ, 'c') or pZ.c.size == 0:
        return True
    
    # If center exists but has zero dimension
    if hasattr(pZ, 'c') and pZ.c.size > 0:
        from .dim import dim
        return dim(pZ) == 0
    
    return False 