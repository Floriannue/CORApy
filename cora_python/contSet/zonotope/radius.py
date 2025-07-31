"""
radius - computes the radius of a hypersphere enclosing a zonotope

Syntax:
    r = radius(Z)

Inputs:
    Z - zonotope object

Outputs:
    r - radius

Example: 
    Z = zonotope([1;-1],[-1 2 1; 3 2 0]);
    r = radius(Z)

Other m-files required: ---
Subfunctions: none
MAT-files required: none

See also: 

Authors:       Matthias Althoff
Written:       19-April-2010
Last update:   27-July-2016
                27-August-2019 (MW)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from .zonotope import Zonotope


def radius(Z: Zonotope) -> float:
    """
    Computes the radius of a hypersphere enclosing a zonotope
    
    Args:
        Z: zonotope object
        
    Returns:
        r: radius
    """
    # Extract generators
    G = Z.G
    
    # Method 1
    # Add length of generators
    r = np.sum(np.linalg.norm(G, axis=0))
    
    # Method 2
    # Convert to interval (axis-aligned box around zonotope)
    IH = Z.interval()
    # Compute half of edge length
    l = IH.rad()
    # Compute enclosing radius
    rAlt = np.linalg.norm(l)
    
    # Choose minimum
    r = min(r, rAlt)
    
    return r 