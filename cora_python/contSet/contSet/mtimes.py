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
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific matrix multiplication logic.
    
    Args:
        M: Matrix or scalar for multiplication
        S: contSet object
        
    Returns:
        ContSet: Result of matrix multiplication
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> S = zonotope([1, 0], [[1, 0], [0, 1]])
        >>> M = np.array([[2, 0], [0, 3]])
        >>> result = mtimes(M, S)  # or result = M * S
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'mtimes not implemented for {type(M).__name__} and {type(S).__name__}') 