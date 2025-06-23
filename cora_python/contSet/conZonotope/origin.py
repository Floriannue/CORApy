"""
origin - instantiates a constrained zonotope representing the origin in R^n

Syntax:
    cZ = origin(n)

Inputs:
    n - dimension

Outputs:
    cZ - conZonotope object representing the origin

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       21-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def origin(n: int) -> 'ConZonotope':
    """
    Instantiates a constrained zonotope representing the origin in R^n
    
    Args:
        n: dimension
        
    Returns:
        cZ: conZonotope object representing the origin
    """
    
    from .conZonotope import ConZonotope
    
    # Create center at origin with no generators
    c = np.zeros((n, 1))
    G = np.zeros((n, 0))
    A = np.zeros((0, 0))
    b = np.zeros((0, 1))
    
    return ConZonotope(c, G, A, b) 