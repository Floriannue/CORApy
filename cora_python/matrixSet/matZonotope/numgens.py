"""
numgens - returns the number of generators of a matZonotope

Syntax:
    N = numgens(matZ)

Inputs:
    matZ - matZonotope object

Outputs:
    N - numeric, number of generators

Example:
    C = np.array([[0, 0], [0, 0]])
    G = np.zeros((2, 2, 2))
    G[:,:,0] = np.array([[1, 3], [-1, 2]])
    G[:,:,1] = np.array([[2, 0], [1, -1]])
    matZ = matZonotope(C, G)
    N = numgens(matZ)

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
    from .matZonotope import matZonotope


def numgens(matZ: 'matZonotope') -> int:
    """
    Returns the number of generators of a matZonotope
    
    Args:
        matZ: matZonotope object
        
    Returns:
        N: number of generators
    """
    
    # G has dimensions (n x m x h)
    # MATLAB: N = size(matZ.G,3);
    if matZ.G.size == 0:
        return 0
    return matZ.G.shape[2] if len(matZ.G.shape) > 2 else 0

