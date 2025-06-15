"""
plus - overloaded '+' operator for the Minkowski addition of two sets or a set with a vector

This function computes the Minkowski addition of two sets or a set with a vector.
The Minkowski addition is defined as: {s_1 + s_2 | s_1 ∈ S_1, s_2 ∈ S_2}

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def plus(S1: Union['ContSet', np.ndarray], S2: Union['ContSet', np.ndarray]) -> 'ContSet':
    """
    Overloaded '+' operator for the Minkowski addition of two sets or a set with a vector
    
    Computes the set {s_1 + s_2 | s_1 ∈ S_1, s_2 ∈ S_2}
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific addition logic.
    
    Args:
        S1: First contSet object or numeric vector
        S2: Second contSet object or numeric vector
        
    Returns:
        ContSet: Result of Minkowski addition
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> S1 = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> S2 = zonotope([0, 1], [[0.5, 0], [0, 0.5]])
        >>> S = plus(S1, S2)  # or S = S1 + S2
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'plus not implemented for {type(S1).__name__} and {type(S2).__name__}') 