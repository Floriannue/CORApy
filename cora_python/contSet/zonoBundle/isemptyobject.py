"""
isemptyobject - checks whether a zonotope bundle contains any information

Syntax:
    res = isemptyobject(zB)

Inputs:
    zB - zonoBundle object

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
    from .zonoBundle import ZonoBundle


def isemptyobject(zB: 'ZonoBundle') -> bool:
    """
    Checks whether a zonotope bundle contains any information
    
    Args:
        zB: zonoBundle object
        
    Returns:
        res: true if the zonotope bundle is empty, false otherwise
    """
    
    # Check if bundle has any zonotopes
    if not hasattr(zB, 'Z') or len(zB.Z) == 0:
        return True
    
    # Check if first zonotope has dimension
    from .dim import dim
    return dim(zB) == 0 