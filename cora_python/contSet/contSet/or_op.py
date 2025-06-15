"""
or - computes an over-approximation for the union of two sets

This function computes an over-approximation of the union of two sets:
{s | s ∈ S₁ ∨ s ∈ S₂}

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def or_op(S1: Union['ContSet', np.ndarray], S2: Union['ContSet', np.ndarray]) -> 'ContSet':
    """
    Computes an over-approximation for the union of two sets
    
    Computes the set {s | s ∈ S₁ ∨ s ∈ S₂}
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific union logic.
    
    Args:
        S1: First contSet object
        S2: Second contSet object
        
    Returns:
        ContSet: Union (over-approximation) of the two sets
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([4, 3], [6, 5])
        >>> result = or_op(S1, S2)  # or result = S1 | S2
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'or not implemented for {type(S1).__name__} and {type(S2).__name__}') 