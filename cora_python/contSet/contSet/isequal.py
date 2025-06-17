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
    
    This implementation uses polymorphic dispatch to call the appropriate
    isequal function based on the type of S1.
    
    Args:
        S1: First contSet object
        S2: Second object to compare with
        tol: Optional tolerance for comparison
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        bool: True if objects are equal, False otherwise
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([1, 2], [3, 4])
        >>> result = isequal(S1, S2)
    """
    # Use polymorphic dispatch
    if hasattr(S1, 'isequal') and callable(getattr(S1, 'isequal')):
        return S1.isequal(S2, tol, *args, **kwargs)
    
    # Fallback - throw error if not implemented
    raise CORAError('CORA:noops', 
                   f'isequal not implemented for {type(S1).__name__} and {type(S2).__name__}') 