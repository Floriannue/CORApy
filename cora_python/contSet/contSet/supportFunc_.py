"""
supportFunc_ - evaluates the support function of a set along a given direction (internal use)

This function provides the internal implementation for support function computation.
It should be overridden in subclasses to provide specific support function logic.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 12-September-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, TYPE_CHECKING
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def supportFunc_(S: 'ContSet', 
                 direction: np.ndarray, 
                 type_: str = 'upper',
                 method: str = 'interval',
                 max_order_or_splits: int = 8,
                 tol: float = 1e-3) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Evaluates the support function of a set along a given direction (internal use)
    
    This function uses polymorphic dispatch to call the appropriate subclass
    implementation of supportFunc_, or provides the base implementation.
    
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
        CORAerror: If supportFunc_ not implemented for the specific set type
        
    Example:
        >>> # This will dispatch to the appropriate subclass implementation
        >>> S = interval([1, 2], [3, 4])
        >>> direction = np.array([[1], [0]])
        >>> val = supportFunc_(S, direction, 'upper')
    """
    # Check if subclass has overridden supportFunc_ method
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'supportFunc_') and 
        base_class and hasattr(base_class, 'supportFunc_') and
        type(S).supportFunc_ is not base_class.supportFunc_):
        return type(S).supportFunc_(direction, type_, method, max_order_or_splits, tol)
    else:
        # Base implementation - throw error as this method should be overridden
        raise CORAerror('CORA:noops',
                       f'supportFunc_ not implemented for {type(S).__name__} with type {type_} and method {method}') 