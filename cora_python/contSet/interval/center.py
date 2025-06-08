"""
center - returns the center of an interval

Syntax:
    c = center(I)

Inputs:
    I - interval object

Outputs:
    c - center of interval (vector)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 26-June-2015 (MATLAB)
Last update: 28-February-2025 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import interval


def center(I: interval) -> np.ndarray:
    """
    Returns the center of an interval
    
    Args:
        I: interval object
        
    Returns:
        c: center of interval (vector)
    """
    # Empty set check
    if I.representsa_('emptySet', 1e-9):
        return np.zeros((I.dim(), 0))
    
    # Fullspace check
    if np.isscalar(I.dim()) and I.representsa_('fullspace', 1e-9):
        return np.zeros(I.dim())
    
    # Compute center
    c = 0.5 * (I.inf + I.sup)
    
    # Handle special case: [-inf, inf] should give 0, not NaN
    if c.ndim == 0:
        # Scalar case
        if np.isinf(I.inf) and np.isinf(I.sup) and I.inf < 0 and I.sup > 0:
            c = np.array(0.0)
        elif np.isinf(c):
            c = np.array(np.nan)
    else:
        # Vector/matrix case
        both_inf_mask = np.isinf(I.inf) & np.isinf(I.sup) & (I.inf < 0) & (I.sup > 0)
        c[both_inf_mask] = 0
        c[np.isinf(c)] = np.nan
    
    return c 