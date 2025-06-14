"""
isequal - checks if two sets are equal

This function provides the core equality comparison for contSet objects.
It is meant to be overridden in subclasses that implement specific equality logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Any, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def isequal(S1: 'ContSet', S2: Any, tol: Optional[float] = None, *args, **kwargs) -> bool:
    """
    Checks if two sets are equal
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific equality logic.
    
    Args:
        S1: First contSet object
        S2: Second object to compare with
        tol: Optional tolerance for comparison
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        bool: True if objects are equal, False otherwise
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes like interval, zonotope, etc.
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([1, 2], [3, 4])
        >>> result = isequal(S1, S2)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops', 
                   f'isequal not implemented for {type(S1).__name__} and {type(S2).__name__}') 