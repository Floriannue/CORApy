"""
isempty - checks if a matrix polytope is empty

Syntax:
   res = isempty(matP)

Inputs:
   matP - matPolytope object

Outputs:
   res - true/false

Example: 
   matP = matPolytope()
   res = isempty(matP)

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
    from .matPolytope import MatPolytope


def isempty(matP: 'MatPolytope') -> bool:
    """
    Checks if a matrix polytope is empty
    
    Args:
        matP: matPolytope object
        
    Returns:
        res: True if empty, False otherwise
    """
    
    return matP.V.size == 0

