"""
isZero - checks if a set represents the origin (DEPRECATED -> representsa)

This function is deprecated and should be replaced with representsa(S, 'origin').

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-July-2023 (MATLAB)
Python translation: 2025
"""

import warnings
from .representsa import representsa


def isZero(S: 'ContSet') -> bool:
    """
    Checks if a set represents the origin (DEPRECATED)
    
    Args:
        S: contSet object
        
    Returns:
        bool: True if set represents the origin, False otherwise
        
    Warning:
        This function is deprecated. Use representsa(S, 'origin') instead.
        
    Example:
        >>> S = interval([0], [0])
        >>> result = isZero(S)  # Deprecated
        >>> result = representsa(S, 'origin')  # Preferred
    """
    warnings.warn(
        "contSet/isZero is deprecated. "
        "Please replace 'isZero(S)' with 'representsa(S, \"origin\")'. "
        "This change was made to unify syntax across all set representations.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return representsa(S, 'origin') 