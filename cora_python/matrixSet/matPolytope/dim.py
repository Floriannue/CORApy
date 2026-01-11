"""
dim - returns the dimension of the matrix polytope

Syntax:
    n = dim(matP)
    n = dim(matP, rc)

Inputs:
    matP - matPolytope object
    rc - 1 for row dimension, 2 for column dimension

Outputs:
    n - array with row and column dimension

Example: 
    V = np.zeros((2, 2, 3))
    V[:,:,0] = np.array([[1, 2], [0, 1]])
    matP = matPolytope(V)
    n = dim(matP)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       03-April-2023 (MATLAB)
Last update:   02-May-2024 (TL, simplified due to new structure of V) (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .matPolytope import MatPolytope


def dim(matP: 'MatPolytope', rc: int = None) -> Union[Tuple[int, int], int]:
    """
    Returns the dimension of the matrix polytope
    
    Args:
        matP: matPolytope object
        rc: 1 for row dimension, 2 for column dimension (optional)
        
    Returns:
        n: array with row and column dimension, or specific dimension if rc specified
    """
    
    # parse input
    if rc is None:
        rc = [1, 2]  # MATLAB: rc = 1:2
    
    if matP.V.size == 0:
        if isinstance(rc, list):
            return (0, 0)
        else:
            return 0
    
    # MATLAB: n = size(matP.V,rc);
    if isinstance(rc, list):
        # Return tuple (rows, columns)
        return (matP.V.shape[0], matP.V.shape[1])
    else:
        # Return specific dimension (1-indexed in MATLAB, 0-indexed in Python)
        if rc == 1:
            return matP.V.shape[0]  # rows
        elif rc == 2:
            return matP.V.shape[1]  # columns
        else:
            raise ValueError("rc must be 1 (rows) or 2 (columns)")

