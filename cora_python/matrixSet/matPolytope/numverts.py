"""
numverts - returns the number of vertices of a matPolytope

Syntax:
    N = numverts(matP)

Inputs:
    matP - matPolytope object

Outputs:
    N - numeric, number of vertices

Example:
    V = np.zeros((2, 2, 3))
    V[:,:,0] = np.array([[1, 2], [0, 1]])
    V[:,:,1] = np.array([[1, 3], [-1, 2]])
    matP = matPolytope(V)
    N = numverts(matP)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       25-April-2024 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matPolytope import MatPolytope


def numverts(matP: 'MatPolytope') -> int:
    """
    Returns the number of vertices of a matPolytope
    
    Args:
        matP: matPolytope object
        
    Returns:
        N: number of vertices
    """
    
    # V has dimensions (n x m x h)
    # MATLAB: N = size(matP.V,3);
    if matP.V.size == 0:
        return 0
    return matP.V.shape[2] if len(matP.V.shape) > 2 else 0

