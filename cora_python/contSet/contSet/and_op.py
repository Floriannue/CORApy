"""
and - overloads '&' operator, computes the intersection of two sets

This function computes the intersection of two sets:
{s | s ∈ S₁, s ∈ S₂}

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-August-2022 (MATLAB)
Last update: 27-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Optional
import numpy as np
from .representsa_ import representsa_
from .and_ import and_
from .reorder import reorder


def and_op(S1: Union['ContSet', np.ndarray], S2: Union['ContSet', np.ndarray, list], 
           method: str = 'exact') -> 'ContSet':
    """
    Overloads '&' operator, computes the intersection of two sets
    
    Computes the set {s | s ∈ S₁, s ∈ S₂}
    
    Args:
        S1: First contSet object or numeric
        S2: Second contSet object, numeric, or cell array
        method: Type of computation ('exact', 'inner', 'outer', 'conZonotope', 'averaging')
        
    Returns:
        ContSet: Intersection of the two sets
        
    Raises:
        ValueError: If dimensions don't match or invalid method
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([2, 1], [4, 3])
        >>> result = and_op(S1, S2)  # or result = S1 & S2
    """
    # Validate method based on set types
    if hasattr(S1, '__class__'):
        class_name = S1.__class__.__name__
        if class_name == 'Ellipsoid':
            if method not in ['inner', 'outer']:
                method = 'outer'  # default for ellipsoid
        elif class_name == 'Zonotope':
            if method not in ['conZonotope', 'averaging']:
                method = 'conZonotope'  # default for zonotope
        else:
            if method not in ['exact']:
                method = 'exact'  # default for other sets
    
    # Order input arguments according to their precedence
    S1, S2 = reorder(S1, S2)
    
    # Check dimension compatibility
    if hasattr(S1, 'dim') and hasattr(S2, 'dim'):
        if S1.dim() != S2.dim():
            raise ValueError(f"Dimension mismatch: S1 has dimension {S1.dim()}, S2 has dimension {S2.dim()}")
    
    try:
        # Call subclass method
        res = and_(S1, S2, method)
    except Exception as ME:
        # Handle empty set cases
        if hasattr(S1, '__class__') and hasattr(S1, 'representsa_'):
            if representsa_(S1, 'emptySet', 1e-15, linearize=0, verbose=1):
                return S1
        elif isinstance(S1, np.ndarray) and S1.size == 0:
            return np.array([])
        
        if hasattr(S2, '__class__') and hasattr(S2, 'representsa_'):
            if representsa_(S2, 'emptySet', 1e-15, linearize=0, verbose=1):
                return S2
        elif isinstance(S2, np.ndarray) and S2.size == 0:
            return np.array([])
        
        raise ME
    
    return res 