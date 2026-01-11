"""
isempty - checks if a matrix zonotope is empty

Syntax:
   res = isempty(matZ)

Inputs:
   matZ - matZonotope object

Outputs:
   res - true/false

Example: 
   matZ = matZonotope()
   res = isempty(matZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def isempty(matZ: 'matZonotope') -> bool:
    """
    Checks if a matrix zonotope is empty
    
    Args:
        matZ: matZonotope object
        
    Returns:
        res: True if empty, False otherwise
    """
    
    return matZ.C.size == 0

