"""
isemptyobject - checks whether a constrained zonotope contains any information

Syntax:
    res = isemptyobject(cZ)

Inputs:
    cZ - conZonotope object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       07-June-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def isemptyobject(cZ: 'ConZonotope') -> bool:
    """
    Checks whether a constrained zonotope contains any information
    
    Args:
        cZ: conZonotope object
        
    Returns:
        res: true if the constrained zonotope is empty, false otherwise
    """
    
    # Check if center and generator matrix are empty
    if not hasattr(cZ, 'c') or cZ.c.size == 0:
        return True
    
    # If center exists but has zero dimension
    if hasattr(cZ, 'c') and cZ.c.size > 0:
        from .dim import dim
        return dim(cZ) == 0
    
    return False 