"""
polyZonotope - converts a zonotope object to a polyZonotope object

Syntax:
    pZ = polyZonotope(Z)

Inputs:
    Z - zonotope object

Outputs:
    pZ - polyZonotope object

Example:
    Z = zonotope([1 2 0 -1;3 1 2 2]);
    pZ = polyZonotope(Z);
    
    figure; xlim([-3,5]); ylim([-3,9]);
    plot(Z,[1,2],'FaceColor','b');
    plot(pZ,[1,2],'r--');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval/polyZonotope, taylm/polyZonotope

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       25-June-2018 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
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