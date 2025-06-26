"""
dim - returns the dimension of the ambient space of a zonotope

Syntax:
    n = dim(Z)

Inputs:
    Z - zonotope object

Outputs:
    n - dimension of the ambient space

Example:
    Z = zonotope([-1, 1, 2], [[2, 4, -3], [2, 1, 0], [0, 2, -1]])
    n = dim(Z)

Other m-files required: zonotope/center
Subfunctions: none
MAT-files required: none

See also: zonotope/rank

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       15-September-2019 (MATLAB)
Last update:   10-January-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np


def dim(Z):
    """
    Returns the dimension of the ambient space of a zonotope
    
    Args:
        Z: zonotope object
        
    Returns:
        int: dimension of the ambient space
    """
    # Simply return the number of rows in the center vector
    # This matches MATLAB: n = size(Z.c,1);
    return Z.c.shape[0] 