"""
dim - returns the dimension of the ambient space of a constrained zonotope

Syntax:
    n = dim(cZ)

Inputs:
    cZ - conZonotope object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def dim(cZ: 'ConZonotope') -> int:
    """
    Returns the dimension of the ambient space of a constrained zonotope
    
    Args:
        cZ: conZonotope object
        
    Returns:
        n: dimension of the ambient space
    """
    
    if hasattr(cZ, 'c') and cZ.c.size > 0:
        if cZ.c.ndim == 1:
            return len(cZ.c)
        else:
            return cZ.c.shape[0]
    elif hasattr(cZ, 'G') and cZ.G.size > 0:
        return cZ.G.shape[0]
    else:
        return 0 