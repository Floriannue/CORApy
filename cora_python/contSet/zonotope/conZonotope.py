"""
conZonotope method for zonotope class
"""

from .zonotope import Zonotope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.conZonotope import ConZonotope


def conZonotope(Z: Zonotope) -> 'ConZonotope':
    """
    Convert a zonotope object into a conZonotope object
    
    Args:
        Z: zonotope object
        
    Returns:
        ConZonotope object
    """
    from ..conZonotope import ConZonotope
    
    # Call constructor with center and generator matrix as inputs
    cZ = ConZonotope(Z.c, Z.G)
    
    return cZ 