"""
generators - returns the generator matrix of a zonotope

Syntax:
    G = generators(Z)

Inputs:
    Z - zonotope object

Outputs:
    G - generator matrix

Example: 
    Z = Zonotope(np.array([[1, 1, 0], [0, 0, 1]]))
    G = generators(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       28-November-2016 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from .zonotope import Zonotope

def generators(Z: Zonotope) -> np.ndarray:
    """
    Returns the generator matrix of a zonotope
    Args:
        Z: zonotope object
    Returns:
        Generator matrix
    """
    return Z.G 