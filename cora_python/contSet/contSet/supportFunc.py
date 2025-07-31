"""
supportFunc - evaluates the support function of a set along a given direction

This function computes max_{x ∈ S} l^T * x for a given direction l.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-August-2022 (MATLAB)
Last update: 27-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union, Tuple, Optional
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def supportFunc(S: 'ContSet', 
                direction: np.ndarray, 
                type_: str = 'upper',
                method: str = 'interval',
                max_order_or_splits: int = 8,
                tol: float = 1e-3) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Evaluates the support function of a set along a given direction
    
    Computes max_{x ∈ S} l^T * x
    
    Args:
        S: contSet object
        direction: Direction vector for which bounds are calculated (n,1)
        type_: Type of computation ('lower', 'upper', 'range')
        method: Method for computation (depends on set type)
        max_order_or_splits: Maximum order or number of splits
        tol: Tolerance for computation
        
    Returns:
        Union[float, Tuple]: Support function value, or tuple of (val, x, fac)
        
    Raises:
        CORAerror: If dimensions don't match or invalid parameters
        ValueError: If invalid type or method
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> direction = np.array([1, 0])
        >>> val = supportFunc(S, direction, 'upper')
    """
    # Validate type
    if type_ not in ['lower', 'upper', 'range']:
        raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")
    
    # Validate method based on set type
    if hasattr(S, '__class__'):
        class_name = S.__class__.__name__
        if class_name == 'PolyZonotope':
            valid_methods = ['interval', 'split', 'bnb', 'bnbAdv', 'globOpt', 'bernstein', 'quadProg']
            if method not in valid_methods:
                raise ValueError(f"Invalid method '{method}' for PolyZonotope. Use one of {valid_methods}.")
        elif class_name == 'ConPolyZono':
            valid_methods = ['interval', 'split', 'conZonotope', 'quadProg']
            if method not in valid_methods:
                raise ValueError(f"Invalid method '{method}' for ConPolyZono. Use one of {valid_methods}.")
    
    # Ensure direction is a column vector
    direction = np.asarray(direction)
    if direction.ndim == 1:
        direction = direction.reshape(-1, 1)
    elif direction.shape[0] == 1 and direction.shape[1] > 1:
        direction = direction.T
    
    # Check dimension compatibility
    if S.dim() != len(direction):
        raise CORAerror('CORA:wrongValue', 
                       f'Direction must be a {S.dim()}-dimensional column vector.')
    
    # Check for zero direction
    if np.allclose(direction, 0):
        raise ValueError("Direction cannot be the zero vector")
    
    # Validate numerical parameters
    if not isinstance(max_order_or_splits, int) or max_order_or_splits <= 0:
        raise ValueError("max_order_or_splits must be a positive integer")
    
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError("tol must be a non-negative number")
    
    try:
        # Call subclass method
        result = S.supportFunc_(direction, type_, method, max_order_or_splits, tol)
        
        # Handle different return types
        if isinstance(result, tuple):
            # Return the first element (the value) for single return
            return result[0]
        else:
            # If result is not a tuple, return it directly
            return result
            
    except Exception as ME:
        # Handle empty set case
        if S.representsa_('emptySet', 1e-15):
            if type_ == 'upper':
                val = float('-inf')
            elif type_ == 'lower':
                val = float('+inf')
            elif type_ == 'range':
                # Return interval(-inf, +inf) - would need interval class
                val = (float('-inf'), float('+inf'))
            
            return val
        else:
            raise ME 