"""
origin - instantiates a probabilistic zonotope representing the origin in R^n

Syntax:
    pZ = origin(n)

Inputs:
    n - dimension

Outputs:
    pZ - probZonotope object representing the origin

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

from cora_python.contSet.probZonotope.probZonotope import ProbZonotope
from cora_python.contSet.zonotope.zonotope import Zonotope

if TYPE_CHECKING:
    pass


def origin(n: int) -> ProbZonotope:
    """
    Instantiates a probabilistic zonotope representing the origin in R^n
    
    Args:
        n: dimension
        
    Returns:
        pZ: probZonotope object representing the origin
    """
    
    # Create origin zonotope and convert to numeric matrix [c, G]
    origin_zono = Zonotope.origin(n)
    if origin_zono.G.size > 0:
        Z_mat = np.hstack([origin_zono.c, origin_zono.G])
    else:
        Z_mat = origin_zono.c

    # Create probabilistic zonotope at origin
    g = np.zeros((n, 0))
    gamma = 2  # Default gamma value
    
    return ProbZonotope(Z_mat, g, gamma) 