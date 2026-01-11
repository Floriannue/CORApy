"""
dim - returns the dimension of the matrix zonotope

Syntax:
    n = dim(matZ)
    n = dim(matZ, rc)

Inputs:
    matZ - matZonotope object
    rc - 1 for row dimension, 2 for column dimension

Outputs:
    n - array with row and column dimension

Example: 
    C = np.array([[0, 0], [0, 0]])
    G = np.zeros((2, 2, 2))
    G[:,:,0] = np.array([[1, 3], [-1, 2]])
    matZ = matZonotope(C, G)
    n = dim(matZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def dim(matZ: 'matZonotope', rc: int = None) -> Union[Tuple[int, int], int]:
    """
    Returns the dimension of the matrix zonotope
    
    Args:
        matZ: matZonotope object
        rc: 1 for row dimension, 2 for column dimension (optional)
        
    Returns:
        n: array with row and column dimension, or specific dimension if rc specified
    """
    
    # parse input
    if rc is None:
        rc = [1, 2]  # MATLAB: rc = 1:2
    
    if matZ.C.size == 0:
        if isinstance(rc, list):
            return (0, 0)
        else:
            return 0
    
    # MATLAB: n = size(matZ.C,rc);
    if isinstance(rc, list):
        # Return tuple (rows, columns)
        return (matZ.C.shape[0], matZ.C.shape[1])
    else:
        # Return specific dimension (1-indexed in MATLAB, 0-indexed in Python)
        if rc == 1:
            return matZ.C.shape[0]  # rows
        elif rc == 2:
            return matZ.C.shape[1]  # columns
        else:
            raise ValueError("rc must be 1 (rows) or 2 (columns)")

