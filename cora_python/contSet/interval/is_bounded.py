"""
is_bounded - check if an interval is bounded

Syntax:
    res = is_bounded(I)

Inputs:
    I - interval object

Outputs:
    res - true if interval is bounded, false otherwise

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval

def is_bounded(I: 'Interval') -> bool:
    """
    Check if interval is bounded
    
    Args:
        I: Interval object
        
    Returns:
        True if interval is bounded (all bounds are finite), False otherwise
    """
    # Empty interval is considered bounded
    if I.inf.size == 0:
        return True
    
    # Check if all bounds are finite
    return np.all(np.isfinite(I.inf)) and np.all(np.isfinite(I.sup)) 
