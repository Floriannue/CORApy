"""
combineVec - returns all possible (cartesian product) combinations of
   arguments (same behavior as of combvec)

Syntax:
    Y = combineVec(*args)

Inputs:
    *args - double arrays

Outputs:
    Y - cartesian product combinations

Example:
    Y = combineVec([1, 2], [3, 4, 5])

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors: ???
         Python translation by AI Assistant
Written: ---
Last update: ---
Last revision: ---
"""

import numpy as np
from typing import List, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def combineVec(*args) -> np.ndarray:
    """
    Returns all possible (cartesian product) combinations of arguments.
    
    Args:
        *args: Variable number of double arrays
        
    Returns:
        Y: Matrix where each column represents one combination
        
    Raises:
        CORAerror: If input arguments are not of type double
    """
    # Parse input
    if not all(isinstance(arg, (np.ndarray, list, int, float)) for arg in args):
        raise CORAerror('CORA:wrongValue', 'some',
                       'All input arguments need to be of type "double"!')
    
    # Convert inputs to numpy arrays and ensure they are 2D
    varargin = []
    for arg in args:
        if isinstance(arg, (int, float)):
            arr = np.array([[arg]])
        else:
            arr = np.array(arg)
            if arr.ndim == 1:
                # Convert to row vector (like MATLAB default)
                arr = arr.reshape(1, -1)
            elif arr.ndim == 0:
                arr = arr.reshape(1, 1)
        varargin.append(arr)
    
    # Trivial cases
    if len(args) == 0:
        return np.array([])
    
    if len(args) == 1:
        return varargin[0]
    
    if len(args) > 2:
        # Recursive call
        return combineVec(varargin[0], combineVec(*varargin[1:]))
    
    # Exactly two inputs
    Y1 = varargin[0]
    Y2 = varargin[1]
    
    # Compute
    n1, N1 = Y1.shape
    n2, N2 = Y2.shape
    
    Y = np.zeros((n1 + n2, N1 * N2))
    
    # MATLAB uses 1-based indexing: for i=1:N2, for j=1:N1
    # Column index: (i-1)*N1 + j
    for i in range(1, N2 + 1):  # 1 to N2 (MATLAB style)
        for j in range(1, N1 + 1):  # 1 to N1 (MATLAB style)
            col_idx = (i - 1) * N1 + j - 1  # Convert to 0-based for Python
            Y[:, col_idx] = np.concatenate([Y1[:, j - 1], Y2[:, i - 1]])
    
    return Y 