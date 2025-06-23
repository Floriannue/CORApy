"""
dim - returns the dimension of the ambient space of a zonotope bundle

Syntax:
    n = dim(zB)

Inputs:
    zB - zonoBundle object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       14-September-2006 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zonoBundle import ZonoBundle


def dim(zB: 'ZonoBundle') -> int:
    """
    Returns the dimension of the ambient space of a zonotope bundle
    
    Args:
        zB: zonoBundle object
        
    Returns:
        n: dimension of the ambient space
    """
    
    # Check if there are zonotopes in the bundle
    if hasattr(zB, 'Z') and len(zB.Z) > 0:
        # Get dimension from first zonotope
        first_zono = zB.Z[0]
        if hasattr(first_zono, 'dim') and callable(first_zono.dim):
            return first_zono.dim()
        elif hasattr(first_zono, 'c') and first_zono.c.size > 0:
            if first_zono.c.ndim == 1:
                return len(first_zono.c)
            else:
                return first_zono.c.shape[0]
    
    return 0 