"""
origin - instantiates a polynomial zonotope representing the origin in R^n

Syntax:
    pZ = origin(n)

Inputs:
    n - dimension

Outputs:
    pZ - polyZonotope object representing the origin

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       21-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def origin(n: int) -> 'PolyZonotope':
    """
    Instantiates a polynomial zonotope representing the origin in R^n
    
    Args:
        n: dimension
        
    Returns:
        pZ: polyZonotope object representing the origin
    """
    
    from .polyZonotope import PolyZonotope
    
    # Create center at origin with no generators
    c = np.zeros((n, 1))
    G = np.zeros((n, 0))
    Grest = np.zeros((n, 0))
    expMat = np.zeros((0, 0))
    id = np.array([])
    
    return PolyZonotope(c, G, Grest, expMat, id) 