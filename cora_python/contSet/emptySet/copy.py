"""
copy - copies the emptySet object (used for dynamic dispatch)

Syntax:
    O_out = copy(O)

Inputs:
    O - emptySet object

Outputs:
    O_out - copied emptySet object

Example: 
    O = EmptySet(2)
    O_out = copy(O)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet


def copy(O: 'EmptySet') -> 'EmptySet':
    """
    Copies the emptySet object (used for dynamic dispatch)
    
    Args:
        O: emptySet object
        
    Returns:
        O_out: copied emptySet object
    """
    # Call copy constructor
    from .emptySet import EmptySet
    O_out = EmptySet(O)
    return O_out 