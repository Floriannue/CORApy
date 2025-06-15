"""
empty - instantiates an empty capsule

Syntax:
   C = capsule.empty(n)

Inputs:
   n - dimension

Outputs:
   C - empty capsule

Example: 
   C = capsule.empty(2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 09-January-2024 (MATLAB)
Last update: 15-January-2024 (TL, parse input)
Python translation: 2025
"""

import numpy as np
from typing import Optional


def empty(n: Optional[int] = None) -> 'Capsule':
    """
    Instantiates an empty capsule
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        Empty capsule object
    """
    from .capsule import Capsule
    
    # Parse input
    if n is None:
        n = 0
    
    if not isinstance(n, int) or n < 0:
        raise ValueError("Dimension must be a non-negative integer")
    
    # Create empty capsule
    return Capsule(np.zeros((n, 0)), np.zeros((n, 0)), np.array([])) 