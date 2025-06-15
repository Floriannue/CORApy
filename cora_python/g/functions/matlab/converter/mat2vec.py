"""
mat2vec - Stores entries of a matrix into a vector

Syntax:
    vec = mat2vec(mat)

Inputs:
    mat - numerical matrix

Outputs:
    vec - numerical vector

Example: 
    mat = np.array([[1, 2], [3, 4]])
    vec = mat2vec(mat)  # returns [1, 3, 2, 4] (column-major order)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: vec2mat

Authors:       Matthias Althoff, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       18-June-2010 (MATLAB)
Last update:   20-April-2023 (TL, simplified, MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np


def mat2vec(mat: np.ndarray) -> np.ndarray:
    """
    Stores entries of a matrix into a vector
    
    Args:
        mat: numerical matrix
        
    Returns:
        vec: numerical vector (column-major order)
    """
    mat = np.asarray(mat)
    
    # reshape to column vector (Fortran/column-major order to match MATLAB)
    vec = mat.reshape(-1, 1, order='F')
    
    return vec 