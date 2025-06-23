"""
and_ - overloads '&' operator, computes the intersection of two sets (internal use)

This function provides the internal implementation for set intersection.
It should be overridden in subclasses to provide specific intersection logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def and_(S1: Union['ContSet', np.ndarray], S2: Union['ContSet', np.ndarray], 
         method: str = 'exact') -> 'ContSet':
    """
    Overloads '&' operator, computes the intersection of two sets (internal use)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of and_, or provides the base implementation.
    
    Args:
        S1: First contSet object
        S2: Second contSet object
        method: Method for intersection computation
        
    Returns:
        ContSet: Intersection of the two sets
        
    Raises:
        CORAerror: If and_ is not implemented for the specific set type
        
    Example:
        >>> # This will dispatch to the appropriate subclass implementation
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([2, 1], [4, 3])
        >>> result = and_(S1, S2, 'exact')
    """
    # Check if subclass has overridden and_ method
    base_class = type(S1).__bases__[0] if type(S1).__bases__ else None
    if (hasattr(type(S1), 'and_') and 
        base_class and hasattr(base_class, 'and_') and
        type(S1).and_ is not base_class.and_):
        return type(S1).and_(S2, method)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror('CORA:noops',
                       f'and_ not implemented for {type(S1).__name__} and {type(S2).__name__} with method {method}') 