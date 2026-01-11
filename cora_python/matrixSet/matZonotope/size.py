"""
size - returns the dimension of the matrix zonotope

Syntax:
    n = size(matZ)
    n = size(matZ, dim)

Inputs:
    matZ - matZonotope object
    dim - specified dimension (optional)

Outputs:
    n - dimension of the matrix zonotope

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       17-January-2023 (MATLAB)
Python translation: 2025
"""

from typing import Union, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def size(matZ: 'matZonotope', dim: int = None) -> Union[Tuple[int, int], int]:
    """
    Returns the dimension of the matrix zonotope
    
    Args:
        matZ: matZonotope object
        dim: specified dimension (optional)
        
    Returns:
        n: dimension of the matrix zonotope
    """
    
    if dim is None:
        # MATLAB: [varargout{1:nargout}] = size(matZ.C);
        return matZ.C.shape
    else:
        # MATLAB: [varargout{1:nargout}] = size(matZ.C,varargin{1});
        if dim < 0 or dim >= len(matZ.C.shape):
            raise IndexError(f"Dimension {dim} out of range for shape {matZ.C.shape}")
        return matZ.C.shape[dim]

