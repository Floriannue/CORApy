"""
quadMap - computes the quadratic map of a set

This function computes the quadratic map of sets:
{ x | x_i = s₁ᵀ Qᵢ s₂, s₁ ∈ S₁, s₂ ∈ S₂, i = 1...w }
where Qᵢ ∈ ℝⁿˣⁿ

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import List, Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def quadMap(S1: 'ContSet', S2: 'ContSet', Q: List[np.ndarray]) -> 'ContSet':
    """
    Computes the quadratic map of a set
    
    Computes the set { x | x_i = s₁ᵀ Qᵢ s₂, s₁ ∈ S₁, s₂ ∈ S₂, i = 1...w }
    where Qᵢ ∈ ℝⁿˣⁿ
    
    Args:
        S1: First contSet object
        S2: Second contSet object
        Q: Quadratic coefficients as a list of matrices
        
    Returns:
        ContSet: Quadratically mapped set
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([0, 1], [2, 3])
        >>> Q = [np.eye(2), np.ones((2, 2))]
        >>> S_quad = quadMap(S1, S2, Q)
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'quadMap not implemented for {type(S1).__name__} and {type(S2).__name__}') 