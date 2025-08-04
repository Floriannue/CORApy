"""
conPolyZono - converts a zonotope to a constrained polynomial zonotope

Syntax:
   cPZ = conPolyZono(Z)

Inputs:
   Z - zonotope object

Outputs:
   cPZ - conPolyZono object

Example: 
   Z = Zonotope(np.array([[-1], [1]]), np.array([[1, 3, 2, 4], [3, 2, 0, 1]]))
   cPZ = conPolyZono(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope, polyZonotope

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 21-January-2020 (MATLAB)
Last update: --- (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from .zonotope import Zonotope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.conPolyZono import ConPolyZono


def conPolyZono(Z: Zonotope) -> 'ConPolyZono':
    """
    Converts a zonotope to a constrained polynomial zonotope
    
    Args:
        Z: zonotope object
        
    Returns:
        ConPolyZono object
        
    Example:
        Z = Zonotope(np.array([[-1], [1]]), np.array([[1, 3, 2, 4], [3, 2, 0, 1]]))
        cPZ = conPolyZono(Z)
    """
    from ..polyZonotope import PolyZonotope
    from ..conPolyZono import ConPolyZono
    
    # First convert to polyZonotope
    pZ = PolyZonotope(Z)
    
    # Then convert to conPolyZono
    cPZ = ConPolyZono(pZ)
    
    return cPZ 