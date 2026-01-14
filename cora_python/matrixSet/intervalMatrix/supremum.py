"""
supremum - returns the supremum of the interval matrix; mainly
    implemented so that supremum can be used for both interval and
    intervalMatrix objects without the need for case differentiation

Syntax:
    M = supremum(intMat)

Inputs:
    intMat - intervalMatrix object

Outputs:
    M - matrix representing the supremum

Example:
    intMat = IntervalMatrix(np.array([[2, 3], [1, 2]]), np.array([[1, 0], [1, 1]]))
    M = supremum(intMat)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: intervalMatrix/infimum

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       16-December-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def supremum(intMat: 'IntervalMatrix'):
    """
    Returns the supremum of the interval matrix
    
    Args:
        intMat: intervalMatrix object
        
    Returns:
        M: matrix representing the supremum
    """
    from cora_python.contSet.interval.supremum import supremum as interval_supremum
    
    return interval_supremum(intMat.int)
