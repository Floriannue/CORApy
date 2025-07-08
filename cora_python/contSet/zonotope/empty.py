"""
empty - instantiates an empty zonotope

Syntax:
    Z = zonotope.empty(n)

Inputs:
    n - dimension

Outputs:
    Z - empty zonotope

Example:
    Z = zonotope.empty(2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       09-January-2024 (MATLAB)
Last update:   15-January-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def empty(n=0):
    """
    Instantiates an empty zonotope
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        zonotope: empty zonotope
    """
    from .zonotope import Zonotope
    
    # Parse input - ensure n is non-negative scalar
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise CORAerror('CORA:wrongInputInConstructor',
                      'Dimension must be a non-negative integer')
    
    # Create a zonotope with empty center and generators to represent empty set
    # This matches MATLAB: zonotope(zeros(n,0))
    if n == 0:
        empty_zono = Zonotope(np.array([]), np.array([]))
    else:
        # Create empty center (n x 0 matrix) and empty generators (n x 0 matrix)
        empty_zono = Zonotope(np.zeros((n, 0)), np.zeros((n, 0)))
    return empty_zono 