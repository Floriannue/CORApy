"""
norm - compute the norm of a set

This function computes the norm of a contSet object using various norm types
and modes. It handles different set types and provides appropriate error handling.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-August-2022 (MATLAB)
Last update: 27-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, Optional
import numpy as np
from .representsa_ import representsa_
from .norm_ import norm_


def norm(S: 'ContSet', norm_type: Union[int, float, str] = 2, mode: str = 'ub') -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute the norm of a set
    
    Args:
        S: contSet object
        norm_type: Norm type (1, 2, inf, or 'fro' for Frobenius)
        mode: Mode for computation ('exact', 'ub', 'ub_convex')
        
    Returns:
        Union[float, Tuple[float, np.ndarray]]: Norm value, or tuple of (norm, point) if point is requested
        
    Raises:
        ValueError: If invalid norm type or mode is provided
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> norm_val = norm(S, 2, 'ub')
        >>> # norm_val is the 2-norm upper bound of the interval
    """
    # Validate mode
    if mode not in ['exact', 'ub', 'ub_convex']:
        raise ValueError(f"Invalid mode '{mode}'. Use 'exact', 'ub', or 'ub_convex'.")
    
    # Validate norm type
    if isinstance(norm_type, str):
        if norm_type != 'fro':
            raise ValueError(f"Invalid string norm type '{norm_type}'. Use 'fro' for Frobenius norm.")
    elif isinstance(norm_type, (int, float)):
        if norm_type not in [1, 2, float('inf')]:
            raise ValueError(f"Invalid numeric norm type {norm_type}. Use 1, 2, or inf.")
    else:
        raise ValueError(f"Norm type must be numeric (1, 2, inf) or string ('fro'), got {type(norm_type)}")
    
    try:
        # Call subclass method
        result = norm_(S, norm_type, mode)
        return result
    except Exception as ME:
        # Empty set case
        if representsa_(S, 'emptySet', 1e-15):
            return float('-inf'), np.array([])
        else:
            raise ME 