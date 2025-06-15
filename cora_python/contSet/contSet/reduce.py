"""
reduce - reduces the order of a set

This function reduces the order of a set representation to make it more compact
while preserving the essential properties of the set.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def reduce(S: 'ContSet', *args, **kwargs) -> 'ContSet':
    """
    Reduces the order of a set
    
    Args:
        S: contSet object
        *args: Additional arguments (depends on set type)
        **kwargs: Additional keyword arguments
        
    Returns:
        ContSet: Reduced set
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = zonotope(center, generators)
        >>> S_reduced = reduce(S, 'girard', 10)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'reduce not implemented for {type(S).__name__} with args {args}') 