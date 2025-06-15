"""
volume - computes the volume of a set

This function computes the volume of a contSet object using various methods.
It handles different set types and provides appropriate error handling.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 18-August-2022 (MATLAB)
Last update: 27-March-2023 (MATLAB)
Python translation: 2025
"""

from typing import Optional, Union
from .representsa_ import representsa_
from .volume_ import volume_


def volume(S: 'ContSet', method: str = 'exact', order: int = 5) -> float:
    """
    Computes the volume of a set
    
    Args:
        S: contSet object
        method: Method for evaluation ('exact', 'reduce', 'alamo')
        order: Zonotope order (for zonotope volume computation)
        
    Returns:
        float: Volume of the set
        
    Raises:
        ValueError: If invalid method is provided
        
    Example:
        >>> S = interval([1, 2], [3, 4])
        >>> vol = volume(S)
        >>> # vol is the volume of the interval
    """
    # Validate input arguments
    if method not in ['exact', 'reduce', 'alamo']:
        raise ValueError(f"Invalid method '{method}'. Use 'exact', 'reduce', or 'alamo'.")
    
    if not isinstance(order, int) or order <= 0:
        raise ValueError(f"Order must be a positive integer, got {order}")
    
    try:
        # Call subclass method
        res = volume_(S, method, order)
    except Exception as ME:
        # Empty set case
        if representsa_(S, 'emptySet', 1e-15):
            res = 0.0
        else:
            raise ME
    
    return res 