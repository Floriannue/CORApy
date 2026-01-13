"""
center - Returns the center of an intervalMatrix

Syntax:
    c = center(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    c - center of the interval matrix

Example:
    M = IntervalMatrix(np.eye(2), 2*np.eye(2))
    c = M.center()

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
Written:       06-May-2021 (MATLAB)
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def center(intMat: 'IntervalMatrix') -> np.ndarray:
    """
    Returns the center of an intervalMatrix
    
    Args:
        intMat: IntervalMatrix object
        
    Returns:
        c: center of the interval matrix
    """
    return intMat.int.center()
