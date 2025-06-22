"""
display - displays the properties of an emptySet object (dimension) on
    the command window

Syntax:
    display(O)

Inputs:
    O - emptySet object

Outputs:
    -

Example: 
    O = EmptySet(2)
    display(O)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Last update:   05-April-2023 (MW, simplify, MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet


def display(O: 'EmptySet') -> str:
    """
    Displays the properties of an emptySet object
    
    Args:
        O: emptySet object
        
    Returns:
        str: formatted display string
    """
    return f"emptySet:\n- dimension: {O.dimension}\n- {O.dimension}-dimensional empty set" 