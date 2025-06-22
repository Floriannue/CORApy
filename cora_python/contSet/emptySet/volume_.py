"""
volume_ - returns the volume of a empty set

Syntax:
    val = volume_(O)

Inputs:
    O - emptySet object

Outputs:
    val - volume

Example: 
    O = EmptySet(2)
    val = volume_(O)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/volume

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Last update:   05-April-2023 (rename volume_) (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet


def volume_(O: 'EmptySet') -> float:
    """
    Returns the volume of an empty set
    
    Args:
        O: emptySet object
        
    Returns:
        val: volume (always 0)
    """
    # volume always zero
    return 0.0 