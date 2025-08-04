"""
copy - copies the zonotope object (used for dynamic dispatch)

Syntax:
    Z_out = copy(Z)

Inputs:
    Z - zonotope object

Outputs:
    Z_out - copied zonotope object

Example: 
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0, -1], [0, 1, 1]]))
    Z_out = copy(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2024 (MATLAB)
Last update: --- (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
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
    return Zonotope(Z)