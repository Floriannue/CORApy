"""
zonoBundle method for zonotope class
"""

from .zonotope import Zonotope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle import ZonoBundle


def zonoBundle(Z: Zonotope) -> 'ZonoBundle':
    """
    Convert a zonotope object into a zonotope bundle object
    
    Args:
        Z: zonotope object
        
    Returns:
        ZonoBundle object
    """
    from ..zonoBundle import ZonoBundle
    
    # Create zonoBundle with a single zonotope
    zB = ZonoBundle([Z])
    
    return zB 