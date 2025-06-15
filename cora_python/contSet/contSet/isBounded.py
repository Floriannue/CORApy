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
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific boundedness checking logic.
    
    Args:
        S: contSet object to check
        
    Returns:
        bool: True if the set is bounded, False otherwise
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> result = isBounded(S)
        >>> # result is True for zonotopes
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'isBounded not implemented for {type(S).__name__}') 