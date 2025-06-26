"""
dim - returns the dimension of the ambient space of a polynomial zonotope

Syntax:
    n = dim(pZ)

Inputs:
    pZ - polyZonotope object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-March-2018 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def dim(pZ: 'PolyZonotope') -> int:
    """
    Returns the dimension of the ambient space of a polynomial zonotope
    
    Args:
        pZ: polyZonotope object
        
    Returns:
        n: dimension of the ambient space
    """
    
    # Simply return the number of rows in the center vector
    # This matches MATLAB: n = size(pZ.c,1);
    return pZ.c.shape[0] 