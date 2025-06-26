"""
empty - instantiates an empty ellipsoid

Syntax:
    E = Ellipsoid.empty(n)

Inputs:
    n - dimension

Outputs:
    E - empty ellipsoid

Example: 
    E = Ellipsoid.empty(2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       09-January-2024 (MATLAB)
Last update:   15-January-2024 (TL, parse input, MATLAB)
Python translation: 2025
"""

import numpy as np

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


def empty(n: int = 0):
    """
    Instantiates an empty ellipsoid
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        E: empty ellipsoid
    """
    # Import here to avoid circular import
    from .ellipsoid import Ellipsoid
    
    # Parse input - n should be nonnegative (0 or positive)
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # For empty ellipsoids:
    # - Q should be zeros(0,0) to indicate empty shape matrix
    # - q should be zeros(n,0) to indicate empty center in n-dimensional space
    # This matches MATLAB behavior where empty sets have specific representations
    return Ellipsoid(np.zeros((0, 0)), np.zeros((n, 0))) 