"""
origin - instantiates a zonotope that contains only the origin

Syntax:
    Z = zonotope.origin(n)

Inputs:
    n - dimension (integer, >= 1)

Outputs:
    Z - zonotope representing the origin

Example:
    Z = zonotope.origin(2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       21-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np


def origin(n):
    """
    Instantiates a zonotope that contains only the origin
    
    Args:
        n: dimension (integer, >= 1)
        
    Returns:
        zonotope: zonotope representing the origin
    """
    from .zonotope import Zonotope
    
    # Input validation
    if not isinstance(n, (int, np.integer)) or n < 1:
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
        raise CORAError('CORA:wrongInputInConstructor',
                      'Dimension must be a positive integer')
    
    return Zonotope(np.zeros((n, 1)), np.zeros((n, 0))) 