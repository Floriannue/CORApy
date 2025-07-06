"""
isIntersecting - checks if two sets intersect

This function checks whether two continuous sets have a non-empty intersection.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-August-2022 (MATLAB)
Last update: 20-September-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def isIntersecting(S1: Union['ContSet', np.ndarray], 
                   S2: Union['ContSet', np.ndarray], 
                   type_: str = 'exact',
                   tol: float = 1e-8) -> bool:
    """
    Checks if two sets intersect
    
    Args:
        S1: First contSet object
        S2: Second contSet object or numeric
        type_: Type of check ('exact', 'approx')
        tol: Tolerance for computation
        
    Returns:
        bool: True if sets intersect, False otherwise
        
    Raises:
        ValueError: If invalid type or dimension mismatch
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([2.5, 3], [4.5, 5])
        >>> result = isIntersecting(S1, S2, 'exact')
        >>> # result is True since intervals overlap
    """
    # Validate type
    if type_ not in ['exact', 'approx']:
        raise ValueError(f"Invalid type '{type_}'. Use 'exact' or 'approx'.")
    
    # Validate tolerance
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError("tol must be a non-negative number")
    
    # Right order of objects
    S1, S2 = S1.reorder(S2)
    
    # Check dimension compatibility
    if hasattr(S1, 'dim') and hasattr(S2, 'dim'):
        if S1.dim() != S2.dim():
            raise CORAerror('CORA:dimensionMismatch', f"Dimension mismatch: S1 has dimension {S1.dim()}, S2 has dimension {S2.dim()}")
    
    try:
        # Call subclass method
        res = S1.isIntersecting_(S2, type_, tol)
    except Exception as ME:
        # Handle empty set case
        if (hasattr(S1, 'representsa_') and S1.representsa_('emptySet', 1e-8)) or \
           (hasattr(S2, 'representsa_') and S2.representsa_('emptySet', 1e-8)):
            res = False
        else:
            raise ME
    
    return res 