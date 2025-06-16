"""
copy - copies the contSet object (used for dynamic dispatch)

This function creates a copy of a contSet object. It should be overridden
in subclasses to provide specific copying logic.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2024 (MATLAB)
Python translation: 2025
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def copy(S: 'ContSet') -> 'ContSet':
    """
    Copies the contSet object (used for dynamic dispatch)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific copying logic.
    
    Args:
        S: contSet object to copy
        
    Returns:
        ContSet: Copied contSet object
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0, -1], [0, 1, 1]])
        >>> S_out = copy(S)
    """
    # Check if the object has a copy method and use it
    if hasattr(S, 'copy') and callable(getattr(S, 'copy')):
        return S.copy()
    
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:notSupported',
                   'The chosen subclass of contSet does not support a dynamic copy operation.') 