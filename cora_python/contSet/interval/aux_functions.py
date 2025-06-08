"""
aux_functions - auxiliary functions for interval operations

This module contains shared utility functions used across multiple interval methods.
Following the MATLAB pattern of aux_ functions.

Authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import interval


def _within_tol(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Check if values are within tolerance
    
    Args:
        a: First array
        b: Second array  
        tol: Tolerance value
        
    Returns:
        Boolean array indicating which elements are within tolerance
    """
    return np.abs(a - b) <= tol


def _reorder_numeric(obj1, obj2):
    """
    Reorder objects to ensure interval comes first
    
    Args:
        obj1: First object
        obj2: Second object
        
    Returns:
        Tuple with interval object first
    """
    # Import here to avoid circular imports
    from .interval import interval
    
    if isinstance(obj1, interval):
        return obj1, obj2
    else:
        return obj2, obj1


def _equal_dim_check(obj1, obj2):
    """
    Check if two objects have equal dimensions
    
    Args:
        obj1: First object
        obj2: Second object
        
    Raises:
        CORAError: If dimensions don't match
    """
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
    
    if hasattr(obj1, 'inf') and hasattr(obj2, 'inf'):
        if obj1.inf.shape != obj2.inf.shape:
            raise CORAError('CORA:dimensionMismatch', 
                           f'Dimension mismatch: {obj1.inf.shape} vs {obj2.inf.shape}')


def _representsa(obj, set_type: str, tol: float = 1e-9) -> bool:
    """
    Check if an object represents a specific set type
    
    Args:
        obj: Object to check
        set_type: Type of set to check ('emptySet', 'origin', 'point')
        tol: Tolerance for comparison
        
    Returns:
        True if object represents the specified type
    """
    if hasattr(obj, 'representsa_'):
        return obj.representsa_(set_type, tol)
    elif set_type == 'emptySet':
        # For numeric objects, they're never empty sets
        return False
    else:
        return False 