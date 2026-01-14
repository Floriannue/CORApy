"""
infimum - returns the infimum of the interval matrix; mainly implemented
    so that infimum can be used for both interval and intervalMatrix
    objects without the need for case differentiation

Syntax:
    M = infimum(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    M - matrix representing the infimum

Example:
    intMat = IntervalMatrix(np.array([[2, 3], [1, 2]]), np.array([[1, 0], [1, 1]]))
    M = infimum(intMat)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       16-December-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def infimum(intMat: 'IntervalMatrix'):
    """
    Returns the infimum of the interval matrix
    
    Args:
        intMat: intervalMatrix object
        
    Returns:
        M: matrix representing the infimum
    """
    from cora_python.contSet.interval.infimum import infimum as interval_infimum
    
    return interval_infimum(intMat.int)
