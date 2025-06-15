"""
convHull_ - computes an enclosure for the convex hull of a set and another set or a point (internal use)

This function provides the internal implementation for convex hull computation.
It should be overridden in subclasses to provide specific convex hull logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Last update: 30-September-2024 (MATLAB)
Python translation: 2025
"""

from typing import Union, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def convHull_(S1: Union['ContSet', np.ndarray], 
              S2: Optional[Union['ContSet', np.ndarray]] = None, 
              method: str = 'exact') -> 'ContSet':
    """
    Computes an enclosure for the convex hull of a set and another set or a point (internal use)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific convex hull logic.
    
    Args:
        S1: First contSet object
        S2: Second contSet object or numeric (optional)
        method: Method for computation
        
    Returns:
        ContSet: Convex hull of the sets
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([4, 3], [6, 5])
        >>> result = convHull_(S1, S2, 'exact')
    """
    # This is overridden in subclass if implemented; throw error
    if S2 is None:
        raise CORAError('CORA:noops',
                       f'convHull_ not implemented for single argument {type(S1).__name__}')
    else:
        raise CORAError('CORA:noops',
                       f'convHull_ not implemented for {type(S1).__name__} and {type(S2).__name__} with method {method}') 