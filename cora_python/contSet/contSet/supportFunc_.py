"""
supportFunc_ - evaluates the support function of a set along a given direction (internal use)

This function provides the internal implementation for support function computation.
It should be overridden in subclasses to provide specific support function logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def supportFunc_(S: 'ContSet', 
                 direction: np.ndarray, 
                 type_: str = 'upper',
                 method: str = 'interval',
                 max_order_or_splits: int = 8,
                 tol: float = 1e-3) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Evaluates the support function of a set along a given direction (internal use)
    
    This is the base implementation that throws an error. Subclasses should
    override this method to provide specific support function logic.
    
    Args:
        S: contSet object
        direction: Column vector specifying the direction
        type_: Type of computation ('lower', 'upper', 'range')
        method: Method for computation
        max_order_or_splits: Maximum order or number of splits
        tol: Tolerance for computation
        
    Returns:
        Union[float, Tuple]: Support function value or tuple of (val, x, fac)
        
    Raises:
        CORAError: Always raised as this method should be overridden in subclasses
        
    Example:
        >>> # This will be overridden in specific set classes
        >>> S = interval([1, 2], [3, 4])
        >>> direction = np.array([[1], [0]])
        >>> val = supportFunc_(S, direction, 'upper')
    """
    # This is overridden in subclass if implemented; throw error
    raise CORAError('CORA:noops',
                   f'supportFunc_ not implemented for {type(S).__name__} with type {type_} and method {method}') 