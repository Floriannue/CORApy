"""
origin - instantiates an ellipsoid that contains only the origin

Syntax:
    E = Ellipsoid.origin(n)

Inputs:
    n - dimension (integer, >= 1)

Outputs:
    E - ellipsoid representing the origin

Example: 
    E = Ellipsoid.origin(2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       21-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from .ellipsoid import Ellipsoid

def origin(n: int):
    """
    Instantiates an ellipsoid that contains only the origin
    Args:
        n: dimension (integer, >= 1)
    Returns:
        E: ellipsoid representing the origin
    """
    # Validate input: must be positive integer
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'positive', 'integer']]])
    return Ellipsoid(np.zeros((n, n)), np.zeros((n, 1))) 