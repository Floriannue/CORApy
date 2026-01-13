"""
rad - returns the radius of an intervalMatrix

Syntax:
    res = rad(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    res - numerical value (matrix)

Example: 
    M = IntervalMatrix(np.eye(2), 2*np.eye(2))
    b = M.rad()

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


def rad(intMat: 'IntervalMatrix') -> np.ndarray:
    """
    Returns the radius of an intervalMatrix
    
    Args:
        intMat: IntervalMatrix object
        
    Returns:
        res: radius of the interval matrix
    """
    return intMat.int.rad()
