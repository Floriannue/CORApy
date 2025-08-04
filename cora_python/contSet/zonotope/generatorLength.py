"""
generatorLength - returns the lengths of the generators

Syntax:
    Glength = generatorLength(Z)

Inputs:
    Z - zonotope object

Outputs:
    Glength - vector of generator length


Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       19-July-2010 (MATLAB)
Last update:   14-March-2019 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from .zonotope import Zonotope


def generatorLength(Z: Zonotope) -> np.ndarray:
    """
    Returns the lengths of the generators
    
    Args:
        Z: zonotope object
        
    Returns:
        Vector of generator lengths
    """
    return np.linalg.norm(Z.G, axis=0) 