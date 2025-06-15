"""
and_ - overloads '&' operator, computes the intersection of two sets (internal use)

This function provides the internal implementation for set intersection.
It should be overridden in subclasses to provide specific intersection logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def and_(S1: Union['ContSet', np.ndarray], S2: Union['ContSet', np.ndarray], 
         method: str = 'exact') -> 'ContSet':
    """
    Overloads '&' operator, computes the intersection of two sets (internal use)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific intersection logic.
    
    Args:
        S1: First contSet object
        S2: Second contSet object
        method: Method for intersection computation
        
    Returns:
        ContSet: Intersection of the two sets
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([2, 1], [4, 3])
        >>> result = and_(S1, S2, 'exact')
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'and_ not implemented for {type(S1).__name__} and {type(S2).__name__} with method {method}') 