"""
shape - returns the shape of an intervalMatrix

Syntax:
    s = shape(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    s - tuple representing the matrix dimensions

Example: 
    M = IntervalMatrix(np.eye(2), 2*np.eye(2))
    s = M.shape()

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def shape(intMat: 'IntervalMatrix') -> Tuple[int, ...]:
    """
    Returns the shape of an intervalMatrix
    
    Args:
        intMat: IntervalMatrix object
        
    Returns:
        s: tuple representing the matrix dimensions
    """
    return intMat.int.shape
