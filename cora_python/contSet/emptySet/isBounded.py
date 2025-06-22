"""
isBounded - determines if a set is bounded

Syntax:
    res = isBounded(O)

Inputs:
    O - emptySet object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       14-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet


def isBounded(O: 'EmptySet') -> bool:
    """
    Determines if a set is bounded
    
    Args:
        O: emptySet object
        
    Returns:
        res: True (empty set is always bounded)
    """
    # Empty set is always bounded
    return True 