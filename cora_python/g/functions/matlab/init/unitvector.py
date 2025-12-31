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
    ---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Tobias Ladner
Written:       27-June-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def unitvector(i: int, n: int) -> np.ndarray:
    """
    Returns the i-th standard unit vector of dimension n
    
    Args:
        i: i-th entry is 1 (1-based index)
        n: dimension
        
    Returns:
        v: standard unit vector (n x 1 numpy array)
    """
    
    # omit checks for performance
    
    # MATLAB: if n==0
    if n == 0:
        # always return empty vector
        # MATLAB: v = [];
        v = np.array([])
    else:
        # init vector of length n
        # MATLAB: v = zeros(n,1);
        v = np.zeros((n, 1))
        # set i-th entry to 1 (MATLAB uses 1-based indexing, Python uses 0-based)
        # MATLAB: v(i) = 1;
        v[i - 1] = 1
    
    return v
