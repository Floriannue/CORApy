"""
dim - returns the dimension of the ambient space of a probabilistic zonotope

Syntax:
    n = dim(pZ)

Inputs:
    pZ - probZonotope object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .probZonotope import ProbZonotope


def dim(pZ: 'ProbZonotope') -> int:
    """
    Returns the dimension of the ambient space of a probabilistic zonotope
    
    Args:
        pZ: probZonotope object
        
    Returns:
        n: dimension of the ambient space
    """
    
    if hasattr(pZ, 'Z') and hasattr(pZ.Z, 'dim'):
        return pZ.Z.dim()
    elif hasattr(pZ, 'Z') and hasattr(pZ.Z, 'c') and pZ.Z.c.size > 0:
        if pZ.Z.c.ndim == 1:
            return len(pZ.Z.c)
        else:
            return pZ.Z.c.shape[0]
    else:
        return 0 