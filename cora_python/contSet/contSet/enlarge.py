"""
enlarge - enlarges the set by the given factor without changing its center

This function enlarges a set by scaling it around its center point.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 11-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def enlarge(S: 'ContSet', factor: np.ndarray) -> 'ContSet':
    """
    Enlarges the set by the given factor without changing its center
    
    Args:
        S: contSet object
        factor: Column vector of factors for the enlargement of each dimension
        
    Returns:
        ContSet: Enlarged set
        
    Raises:
        ValueError: If factor is not a column vector
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> factor = np.array([[2], [1.5]])
        >>> S_enlarged = enlarge(S, factor)
    """
    # Validate input
    factor = np.asarray(factor)
    if factor.ndim == 1:
        factor = factor.reshape(-1, 1)
    elif factor.shape[1] != 1:
        raise ValueError("factor must be a column vector")
    
    # Shift to origin
    c = S.center()
    S_shifted = S - c
    
    # Enlarge set (element-wise multiplication)
    S_enlarged = S_shifted * factor
    
    # Shift back to center
    S_result = S_enlarged + c
    
    return S_result 