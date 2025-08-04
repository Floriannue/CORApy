"""
conZonotope - convert a zonotope object into a conZonotope object

Syntax:
   cZ = conZonotope(Z)

Inputs:
   Z - zonotope object

Outputs:
   cZ - conZonotope object

Example:
   Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
   cZ = conZonotope(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 23-May-2018 (MATLAB)
Last update: --- (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
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