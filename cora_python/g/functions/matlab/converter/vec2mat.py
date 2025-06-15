"""
vec2mat - Stores entries of a vector in a matrix

Syntax:
    mat = vec2mat(vec)
    mat = vec2mat(vec, cols)

Inputs:
    vec - vector
    cols - number of columns for the matrix (optional)

Outputs:
    mat - matrix

Example: 
    vec = np.array([1, 2, 3, 4])
    mat = vec2mat(vec, 2)  # returns [[1, 3], [2, 4]] (column-major order)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mat2vec

Authors:       Matthias Althoff, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       18-June-2010 (MATLAB)
Last update:   22-June-2010 (MATLAB)
               05-October-2010 (MATLAB)
               20-April-2023 (TL, simplified, MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Optional


def vec2mat(vec: np.ndarray, cols: Optional[int] = None) -> np.ndarray:
    """
    Stores entries of a vector in a matrix
    
    Args:
        vec: vector
        cols: number of columns for the matrix (optional)
        
    Returns:
        mat: matrix
    """
    vec = np.asarray(vec).flatten()
    
    # Handle empty vector case
    if len(vec) == 0:
        if cols is None or cols == 0:
            return np.array([]).reshape(0, 0)
        else:
            return np.array([]).reshape(0, cols)
    
    # get cols
    if cols is None:
        # assume square
        cols = int(np.sqrt(len(vec)))
    
    # MATLAB reshape(vec, [], cols) means: automatic rows, specified cols
    # This is equivalent to reshape(-1, cols) in numpy
    mat = vec.reshape(-1, cols, order='F')
    
    return mat 