"""
center - returns the center of a zonotope

Syntax:
    c = center(Z)

Inputs:
    Z - zonotope object

Outputs:
    c - center of the zonotope Z

Example:
    Z = Zonotope([1, 0], [[1, 0], [0, 1]])
    c = center(Z)

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
             14-March-2021 (MW, empty set) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .zonotope import Zonotope


def center(Z: Zonotope) -> np.ndarray:
    """
    Returns the center of a zonotope
    
    Args:
        Z: Zonotope object
        
    Returns:
        c: center of the zonotope Z
    """
    return Z.c 