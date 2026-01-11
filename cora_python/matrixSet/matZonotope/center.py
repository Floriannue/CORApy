"""
center - Returns the center of a matZonotope

Syntax:
    C = center(matZ)

Inputs:
    matZ - matZonotope object

Outputs:
    C - center of the matrix zonotope

Example:
    C = np.eye(2)
    G = np.zeros((2, 2, 2))
    G[:,:,0] = np.eye(2)*2
    G[:,:,1] = np.eye(2)
    matZ = matZonotope(C, G)
    C = center(matZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       23-July-2020 (MATLAB)
Last update:   25-April-2024 (TL, matZ.C property) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def center(matZ: 'matZonotope') -> np.ndarray:
    """
    Returns the center of a matZonotope
    
    Args:
        matZ: matZonotope object
        
    Returns:
        C: center of the matrix zonotope
    """
    
    # MATLAB: C = matZ.C;
    return matZ.C

