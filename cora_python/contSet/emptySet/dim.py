"""
dim - returns the dimension of the ambient space of an empty set

Syntax:
    n = dim(O)

Inputs:
    O - emptySet object

Outputs:
    n - dimension

Example: 
    O = EmptySet(2)
    n = dim(O)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet


def dim(O: 'EmptySet') -> int:
    """
    Returns the dimension of the ambient space of an empty set
    
    Args:
        O: emptySet object
        
    Returns:
        n: dimension
    """
    return O.dimension 