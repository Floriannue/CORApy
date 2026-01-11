"""
size - returns the dimension of the matrix polytope

Syntax:
    n = size(matP)
    n = size(matP, dim)

Inputs:
    matP - matPolytope object
    dim - specified dimension (optional)

Outputs:
    n - dimension of the matrix polytope

Example:
    V = np.zeros((2, 2, 2))
    V[:,:,0] = np.array([[1, 2], [0, 1]])
    V[:,:,1] = np.array([[1, 3], [-1, 2]])
    matP = matPolytope(V)
    n = size(matP)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       17-January-2023 (MATLAB)
Last update:   02-May-2024 (TL, simplified) (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .matPolytope import MatPolytope


def size(matP: 'MatPolytope', dim: int = None) -> Union[Tuple[int, int], int]:
    """
    Returns the dimension of the matrix polytope
    
    Args:
        matP: matPolytope object
        dim: specified dimension (optional)
        
    Returns:
        n: dimension of the matrix polytope
    """
    
    # parse input
    if dim is None:
        # MATLAB: dim = 1:2;
        # MATLAB: [varargout{1:nargout}] = size(matP.V,dim);
        return matP.V.shape[:2] if matP.V.size > 0 else (0, 0)
    else:
        # MATLAB: dim = varargin{1};
        # MATLAB: [varargout{1:nargout}] = size(matP.V,dim);
        if dim < 0 or dim >= len(matP.V.shape):
            raise IndexError(f"Dimension {dim} out of range for shape {matP.V.shape}")
        return matP.V.shape[dim]

