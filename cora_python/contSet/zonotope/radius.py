"""
radius method for zonotope class
"""

import numpy as np
from .zonotope import Zonotope


def radius(Z: Zonotope) -> float:
    """
    Computes the radius of a hypersphere enclosing a zonotope
    
    Args:
        Z: zonotope object
        
    Returns:
        Radius of the enclosing hypersphere
    """
    # Extract generators
    G = Z.G
    
    # Method 1: Add length of generators
    # Sum of norms of all generators
    r = np.sum(np.linalg.norm(G, axis=0))
    
    # Method 2: Convert to interval (axis-aligned box around zonotope)
    IH = Z.interval()
    
    # Compute half of edge length (radius of interval)
    l = IH.rad()
    
    # Compute enclosing radius
    rAlt = np.linalg.norm(l)
    
    # Choose minimum
    r = min(r, rAlt)
    
    return r 