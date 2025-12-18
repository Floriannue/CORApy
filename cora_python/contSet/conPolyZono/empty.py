"""
empty - instantiates an empty constrained polynomial zonotope

Syntax:
    cPZ = conPolyZono.empty(n)

Inputs:
    n - dimension

Outputs:
    cPZ - empty constrained polynomial zonotope

Example: 
    cPZ = conPolyZono.empty(2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       09-January-2024 (MATLAB)
Last update:   15-January-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


def empty(n: int = 0):
    """
    Instantiates an empty constrained polynomial zonotope
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        cPZ: empty conPolyZono object
    """
    from .conPolyZono import ConPolyZono
    
    # Parse input - match MATLAB behavior
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # MATLAB: cPZ = conPolyZono(zeros(n,0));
    # Create empty center (n x 0 matrix) - this will create an empty conPolyZono
    c = np.zeros((n, 0))
    
    # Use normal constructor - it will handle empty inputs correctly
    cPZ = ConPolyZono(c)
    
    return cPZ

