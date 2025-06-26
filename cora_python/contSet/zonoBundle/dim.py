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
    
    # MATLAB logic: if zB.parallelSets == 0 then n = 0; else n = size(zB.Z{1}.c,1);
    if zB.parallelSets == 0:
        # fully-empty
        return 0
    else:
        return zB.Z[0].c.shape[0] 