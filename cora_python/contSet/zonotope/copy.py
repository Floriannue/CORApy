"""
copy - creates a copy of a zonotope object

Syntax:
    Z_copy = copy(Z)
    Z_copy = Z.copy()

Inputs:
    Z - zonotope object

Outputs:
    Z_copy - copy of the zonotope object

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .zonotope import Zonotope

def copy(Z: Zonotope) -> Zonotope:
    """
    Creates a copy of a zonotope object
    
    Args:
        Z: Zonotope object to copy
        
    Returns:
        Zonotope: Copy of the zonotope object
        
    Example:
        >>> Z = Zonotope([1, 0], [[1, 0], [0, 1]])
        >>> Z_copy = copy(Z)
    """
    
    # Create a new zonotope with copied data
    return Zonotope(Z.c.copy(), Z.G.copy()) 