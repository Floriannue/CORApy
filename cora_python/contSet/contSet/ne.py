"""
ne - overloads the '~=' operator

This function implements inequality comparison for contSet objects by negating
the result of isequal.

Authors: Mark Wetzlinger, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 09-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import Any
from .isequal import isequal


def ne(S1: 'ContSet', S2: Any, *args, **kwargs) -> bool:
    """
    Overloads the '~=' operator for contSet objects
    
    Args:
        S1: First contSet object
        S2: Second object to compare with
        *args: Additional arguments passed to isequal
        **kwargs: Additional keyword arguments passed to isequal
        
    Returns:
        bool: True if objects are not equal, False otherwise
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([1, 2], [5, 6])
        >>> result = ne(S1, S2)
        >>> # result is True
    """
    return not isequal(S1, S2, *args, **kwargs) 