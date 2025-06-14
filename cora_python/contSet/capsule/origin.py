"""
origin - instantiates a capsule that contains only the origin

Syntax:
   C = capsule.origin(n)

Inputs:
   n - dimension (integer, >= 1)

Outputs:
   C - capsule representing the origin

Example: 
   C = capsule.origin(2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np


def origin(n: int) -> 'Capsule':
    """
    Instantiates a capsule that contains only the origin
    
    Args:
        n: dimension (integer, >= 1)
        
    Returns:
        Capsule representing the origin
    """
    from .capsule import Capsule
    
    if not isinstance(n, int) or n < 1:
        raise ValueError("Dimension must be a positive integer")
    
    return Capsule(np.zeros((n, 1)), np.zeros((n, 1)), 0) 