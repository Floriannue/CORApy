"""
isBounded - determines if a zonotope is bounded

Syntax:
    res = isBounded(Z)

Inputs:
    Z - zonotope object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 24-July-2023 (MATLAB)
Last update: --- (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zonotope import Zonotope

def isBounded(Z: 'Zonotope') -> bool:
    """
    Determines if a zonotope is bounded
    
    Args:
        Z: Zonotope object
        
    Returns:
        bool: Always True since zonotopes are always bounded
        
    Example:
        >>> Z = Zonotope([1, 0], [[1, 0], [0, 1]])
        >>> result = isBounded(Z)
        >>> # result is True
    """
    # Zonotopes are always bounded
    return True 