"""
unitvector - returns the i-th standard unit vector of dimension n

Syntax:
    v = unitvector(i,n)

Inputs:
    i - i-th entry is 1
    n - dimension

Outputs:
    v - standard unit vector

Example:
    v = unitvector(2, 3)  # returns [0, 1, 0]


See also: ---

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       27-June-2023 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import numpy as np


def unitvector(i: int, n: int) -> np.ndarray:
    """
    Returns the i-th standard unit vector of dimension n
    
    Args:
        i: i-th entry is 1 (1-indexed)
        n: dimension
        
    Returns:
        Standard unit vector as column vector
    """
    # omit checks for performance 
    
    if n == 0:
        # always return empty vector
        v = np.array([]).reshape(0, 1)
    else:
        # init vector of length n
        v = np.zeros((n, 1))
        # set i-th entry to 1 (convert to 0-indexed)
        v[i-1] = 1
    
    return v 