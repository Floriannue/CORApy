"""
isBounded - determines if a set is bounded

This function checks whether a contSet object represents a bounded set.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 24-July-2023 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def isBounded(S: 'ContSet') -> bool:
    """
    Determines if a set is bounded
    
    This function delegates to the object's isBounded method if available,
    otherwise raises an error for the base contSet class.
    
    Args:
        S: contSet object to check
        
    Returns:
        bool: True if the set is bounded, False otherwise
        
    Raises:
        CORAError: If isBounded is not implemented for the specific set type
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> result = isBounded(S)
        >>> # result is True for zonotopes
    """
    # Check if the object has an isBounded method and use it
    if hasattr(S, 'isBounded') and callable(getattr(S, 'isBounded')):
        return S.isBounded()
    
    # Fallback error for base contSet objects
    raise CORAError('CORA:noops',
                   f'isBounded not implemented for {type(S).__name__}') 