"""
conPolyZono method for zonotope class
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