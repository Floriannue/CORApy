"""
mtimes - overloaded '*' operator for the multiplication of a matrix with a set

This function computes the set {M * s | s ∈ S}, where S ∈ ℝⁿ and M ∈ ℝʷˣⁿ.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def mtimes(M: Union[np.ndarray, float, int], S: 'ContSet') -> 'ContSet':
    """
    Overloaded '*' operator for the multiplication of a matrix with a set
    
    Computes the set {M * s | s ∈ S}, where S ∈ ℝⁿ and M ∈ ℝʷˣⁿ
    
    This function delegates to the object's mtimes method if available,
    otherwise raises an error.
    
    Args:
        M: Matrix or scalar for multiplication
        S: contSet object
        
    Returns:
        ContSet: Result of matrix multiplication
        
    Raises:
        CORAError: If mtimes is not implemented for the specific set type
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> M = np.array([[2, 0], [0, 3]])
        >>> result = mtimes(M, S)  # or result = M * S
    """
    # Check if the set object has an mtimes method and use it
    if hasattr(S, 'mtimes') and callable(getattr(S, 'mtimes')):
        return S.mtimes(M)
    
    # Fallback error
    raise CORAError('CORA:noops',
                   f'mtimes not implemented for {type(M).__name__} and {type(S).__name__}') 