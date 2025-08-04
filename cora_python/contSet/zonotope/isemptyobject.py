"""
isemptyobject - checks whether a zonotope contains any information at
    all; consequently, the set is interpreted as the empty set

Syntax:
    res = isemptyobject(Z)

Inputs:
    Z - zonotope object

Outputs:
    res - true/false

Example: 
    Z = Zonotope(np.array([[2], [1]]))
    isemptyobject(Z)  # false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       24-July-2023 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zonotope import Zonotope

def isemptyobject(Z: 'Zonotope') -> bool:
    """
    Checks whether a zonotope contains any information at all
    
    Args:
        Z: zonotope object
        
    Returns:
        bool: True if zonotope is empty, False otherwise
    """
    # A zonotope is empty if BOTH center and generators are empty
    # This matches MATLAB: isnumeric(Z.c) && isempty(Z.c) && isnumeric(Z.G) && isempty(Z.G)
    center_empty = Z.c is None or Z.c.size == 0
    generators_empty = Z.G is None or Z.G.size == 0
    
    return center_empty and generators_empty 