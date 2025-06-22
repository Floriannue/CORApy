"""
isemptyobject - checks whether an empty set contains any information at
    all; consequently, the set is interpreted as the empty set 

Syntax:
    res = isemptyobject(O)

Inputs:
    O - emptySet object

Outputs:
    res - true/false

Example: 
    O = EmptySet(2)
    isemptyobject(O)  # False

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       24-July-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet


def isemptyobject(O: 'EmptySet') -> bool:
    """
    Checks whether an empty set contains any information at all
    
    Args:
        O: emptySet object
        
    Returns:
        res: True/False (always False for emptySet - it represents empty but is not itself empty)
    """
    # In MATLAB, this always returns false for emptySet objects
    # The emptySet object itself is not "empty" - it represents the empty set
    return False 