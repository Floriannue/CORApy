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
    
    This function delegates to the object's plus method if available,
    otherwise raises an error.
    
    Args:
        S1: First contSet object or numeric vector
        S2: Second contSet object or numeric vector
        
    Returns:
        ContSet: Result of Minkowski addition
        
    Raises:
        CORAError: If plus is not implemented for the specific set types
        
    Example:
        >>> S1 = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> S2 = zonotope([0, 1], [[0.5, 0], [0, 0.5]])
        >>> S = plus(S1, S2)  # or S = S1 + S2
    """
    # Check if the first object has a plus method and use it
    if hasattr(S1, 'plus') and callable(getattr(S1, 'plus')):
        return S1.plus(S2)
    
    # Check if the second object has a plus method and use it (commutative)
    if hasattr(S2, 'plus') and callable(getattr(S2, 'plus')):
        return S2.plus(S1)
    
    # Fallback error
    raise CORAError('CORA:noops',
                   f'plus not implemented for {type(S1).__name__} and {type(S2).__name__}') 