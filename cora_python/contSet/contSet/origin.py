"""
origin - instantiates a set representing only the origin

This function creates a set that represents only the origin point in n-dimensional space.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-September-2024 (MATLAB)
Python translation: 2025
"""
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def origin(S: 'ContSet', n: int) -> 'ContSet':
    """
    Instantiates a set representing only the origin
    
    Args:
        n: Dimension of the space
        
    Returns:
        ContSet: Set representing only the origin
        
    Raises:
        CORAerror: Always raised as this method should be overridden in subclasses
        ValueError: If n is not a positive integer
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = origin(3)  # Creates origin in 3D space
    """
    # Validate input
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    # This is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:notSupported',
                   'The chosen subclass of contSet does not support representing only the origin.') 