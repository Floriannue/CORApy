"""
reorder - reorder two sets according to their preference value

This function reorders two sets according to their precedence value.
Numeric types are also reordered to position 2.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 28-September-2024 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple
import numpy as np


def reorder(S1: Union['ContSet', np.ndarray], S2: Union['ContSet', np.ndarray]) -> Tuple[Union['ContSet', np.ndarray], Union['ContSet', np.ndarray]]:
    """
    Reorder two sets according to their preference value
    
    Numeric types are also reordered to position 2.
    
    Args:
        S1: First contSet object or numeric
        S2: Second contSet object or numeric
        
    Returns:
        Tuple: (S1, S2) with S1 having lower precedence, S2 having higher precedence
        
    Example:
        >>> S1 = interval([1, 2], [3, 4])  # precedence = 120
        >>> S2 = zonotope([0, 0], [[1, 0], [0, 1]])  # precedence = 110
        >>> S1_reordered, S2_reordered = reorder(S1, S2)
        >>> # S1_reordered will be the zonotope, S2_reordered will be the interval
    """
    # Classic swap using temporary variable
    if (isinstance(S1, (np.ndarray, list, tuple)) or np.isscalar(S1)) or \
       (hasattr(S2, 'precedence') and hasattr(S1, 'precedence') and 
        S2.precedence < S1.precedence):
        # Swap S1 and S2
        S1, S2 = S2, S1
    
    return S1, S2 