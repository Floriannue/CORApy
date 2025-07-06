"""
dim - returns the dimension of the interval matrix

Syntax:
    n = dim(intMat)
    n = dim(intMat, rc)

Inputs:
    intMat - intervalMatrix object
    rc - 1 for row dimension, 2 for column dimension

Outputs:
    n - array with row and column dimension

Example: 
    intMat = IntervalMatrix(np.array([[2, 3], [1, 2]]), np.array([[1, 0], [1, 1]]))
    n = intMat.dim()

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def dim(intMat: 'IntervalMatrix', rc: int = None) -> Union[Tuple[int, int], int]:
    """
    Returns the dimension of the interval matrix
    
    Args:
        intMat: intervalMatrix object
        rc: 1 for row dimension, 2 for column dimension (optional)
        
    Returns:
        n: array with row and column dimension, or specific dimension if rc specified
    """
    
    # Choose dimension of infimum (like MATLAB)
    if rc is None:
        # Return tuple (rows, columns)
        return intMat.int.inf.shape
    else:
        # Return specific dimension (1-indexed in MATLAB, 0-indexed in Python)
        if rc == 1:
            return intMat.int.inf.shape[0]  # rows
        elif rc == 2:
            return intMat.int.inf.shape[1]  # columns
        else:
            raise ValueError("rc must be 1 (rows) or 2 (columns)") 