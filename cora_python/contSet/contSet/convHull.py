"""
convHull - computes an enclosure for the convex hull of a set and another set or a point

This function computes the convex hull of sets:
{λs₁ + (1-λ)s₂ | s₁,s₂ ∈ S₁ ∪ S₂, λ ∈ [0,1]}

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union, Optional
import numpy as np
from .convHull_ import convHull_
from .reorder import reorder


if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def convHull(S1: Union['ContSet', np.ndarray], 
             S2: Optional[Union['ContSet', np.ndarray, list]] = None, 
             method: str = 'exact') -> 'ContSet':
    """
    Computes an enclosure for the convex hull of a set and another set or a point
    
    Computes the set {λs₁ + (1-λ)s₂ | s₁,s₂ ∈ S₁ ∪ S₂, λ ∈ [0,1]}
    
    Args:
        S1: First contSet object
        S2: Second contSet object, numeric, or cell array (optional)
        method: Method for computation ('exact', 'outer', 'inner')
        
    Returns:
        ContSet: Convex hull of the sets
        
    Raises:
        ValueError: If invalid method or dimension mismatch
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])
        >>> S2 = interval([4, 3], [6, 5])
        >>> result = convHull(S1, S2, 'exact')
    """
    from ..zonotope import Zonotope
    # Special case: single argument
    if S2 is None:
        return convHull_(S1)
    
    # Validate method
    if method not in ['exact', 'outer', 'inner']:
        raise ValueError(f"Invalid method '{method}'. Use 'exact', 'outer', or 'inner'.")
    
    # convert numeric values to zonotope
    if isinstance(S2, (np.ndarray, list)):
        S2 = Zonotope(S2)

    # Order input arguments according to their precedence
    S1, S2 = reorder(S1, S2)
    
    # Check dimension compatibility
    if hasattr(S1, 'dim') and hasattr(S2, 'dim') and callable(getattr(S1, 'dim', None)) and callable(getattr(S2, 'dim', None)):
        if S1.dim() != S2.dim():
            raise ValueError(f"Dimension mismatch: S1 has dimension {S1.dim()}, S2 has dimension {S2.dim()}")
    
    # call subclass method if it exists
    if hasattr(S1, 'convHull_') and callable(getattr(S1, 'convHull_', None)):
        return S1.convHull_(S2, method)
    else:
        # Fallback to general implementation
        return convHull_(S1, S2, method) 