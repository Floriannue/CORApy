"""
delta - returns the width/delta of an intervalMatrix

Syntax:
    res = delta(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    res - numerical value (matrix) - width/delta matrix

Example: 
    M = IntervalMatrix(np.eye(2), 2*np.eye(2))
    d = M.delta()

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def delta(intMat: 'IntervalMatrix') -> np.ndarray:
    """
    Returns the width/delta of an intervalMatrix (alias for rad)
    
    Args:
        intMat: IntervalMatrix object
        
    Returns:
        res: width/delta of the interval matrix
    """
    return intMat.int.rad()
