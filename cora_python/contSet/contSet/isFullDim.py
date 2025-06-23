"""
isFullDim - checks if the dimension of the affine hull of a set is equal to the dimension of its ambient space

This function checks whether a set is full-dimensional in its ambient space.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def isFullDim(S: 'ContSet') -> bool:
    """
    Checks if the dimension of the affine hull of a set is equal to the dimension of its ambient space
    
    Args:
        S: contSet object
        
    Returns:
        bool: True if set is full-dimensional, False otherwise
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = interval([1, 2], [3, 4])
        >>> result = isFullDim(S)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops',
                   f'isFullDim not implemented for {type(S).__name__}') 