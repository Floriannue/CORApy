"""
abs - returns the absolute value bound of an interval matrix

Syntax:
    M = abs(intMat)

Inputs:
    intMat - interval matrix

Outputs:
    M - absolute value bound

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def abs(intMat: 'IntervalMatrix') -> np.ndarray:
    """
    Returns the absolute value bound of an interval matrix
    
    Args:
        intMat: intervalMatrix object
        
    Returns:
        M: Absolute value bound matrix
    """
    # MATLAB: M = supremum(abs(intMat.int));
    M = intMat.int.abs().supremum()
    return M
