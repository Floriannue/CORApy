"""
eq - overloads the '==' operator

This function implements equality comparison for contSet objects by delegating
to the isequal method.

Authors: Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 09-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import Any
from .isequal import isequal


def eq(S1: 'ContSet', S2: Any, *args, **kwargs) -> bool:
    """
    Overloads the '==' operator for contSet objects
    
    Args:
        S1: First contSet object
        S2: Second object to compare with
        *args: Additional arguments passed to isequal
        **kwargs: Additional keyword arguments passed to isequal
        
    Returns:
        bool: True if objects are equal, False otherwise
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([1, 2], [3, 4])
        >>> result = eq(S1, S2)
        >>> # result is True
    """
    return isequal(S1, S2, *args, **kwargs) 