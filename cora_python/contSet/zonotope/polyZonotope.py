"""
polyZonotope method for zonotope class
"""

import numpy as np
from .zonotope import Zonotope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope


def polyZonotope(Z: Zonotope) -> 'PolyZonotope':
    """
    Converts a zonotope object to a polyZonotope object
    
    Args:
        Z: zonotope object
        
    Returns:
        PolyZonotope object
    """
    from ..polyZonotope import PolyZonotope
    
    c = Z.c
    G = Z.G
    E = np.eye(G.shape[1])
    
    # Create polyZonotope with center, dependent generators, no independent generators, and identity exponent matrix
    pZ = PolyZonotope(c, G, None, E)
    
    return pZ 