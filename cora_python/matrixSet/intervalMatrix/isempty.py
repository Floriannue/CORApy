"""
isempty - checks if an interval matrix is empty

Syntax:
   res = isempty(intMat)

Inputs:
   intMat - intervalMatrix object

Outputs:
   res - true/false

Example: 
   intMat = IntervalMatrix(np.array([[2, 3], [1, 2]]), np.array([[1, 0], [1, 1]]))
   res = intMat.isempty()

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def isempty(intMat: 'IntervalMatrix') -> bool:
    """
    Checks if an interval matrix is empty
    
    Args:
        intMat: intervalMatrix object
        
    Returns:
        res: True if empty, False otherwise
    """
    
    # Check dimension of interval matrix
    dimensions = intMat.dim()
    res = any(d == 0 for d in dimensions)
    
    return res 