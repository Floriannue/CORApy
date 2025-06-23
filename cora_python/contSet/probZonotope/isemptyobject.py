"""
isemptyobject - checks whether a probabilistic zonotope contains any information

Syntax:
    res = isemptyobject(pZ)

Inputs:
    pZ - probZonotope object

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
    from .probZonotope import ProbZonotope


def isemptyobject(pZ: 'ProbZonotope') -> bool:
    """
    Checks whether a probabilistic zonotope contains any information
    
    Args:
        pZ: probZonotope object
        
    Returns:
        res: true if the probabilistic zonotope is empty, false otherwise
    """
    
    # Check if underlying zonotope is empty
    if not hasattr(pZ, 'Z'):
        return True
    
    # Use the zonotope's isemptyobject method
    if hasattr(pZ.Z, 'isemptyobject'):
        return pZ.Z.isemptyobject()
    
    # Fallback: check dimension
    if hasattr(pZ, 'dim'):
        return pZ.dim() == 0
    
    return True 