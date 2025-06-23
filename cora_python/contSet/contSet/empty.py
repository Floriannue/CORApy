"""
empty - instantiates an empty set

This static method creates an empty set of the specified dimension.
It should be overridden in subclasses that support empty set instantiation.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 09-January-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def empty(n: int) -> 'ContSet':
    """
    Instantiates an empty set
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific empty set instantiation.
    
    Args:
        n: Dimension of the empty set
        
    Returns:
        ContSet: Empty set object
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes like interval, zonotope, etc.
        >>> S = interval.empty(2)  # Creates 2D empty interval
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:notSupported',
                   'The chosen subclass of contSet does not support an empty set instantiation.') 