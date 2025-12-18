"""
empty - instantiates an empty polynomial zonotope

Syntax:
    pZ = empty(n)

Inputs:
    n - dimension

Outputs:
    pZ - empty polyZonotope object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       09-January-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def empty(n: int = 0) -> 'PolyZonotope':
    """
    Instantiates an empty polynomial zonotope
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        pZ: empty polyZonotope object
    """
    
    from .polyZonotope import PolyZonotope
    from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
    
    # parse input - match MATLAB behavior
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # Create empty polyZonotope like MATLAB: polyZonotope(zeros(n,0))
    # This creates a polyZonotope with empty center (n x 0 matrix)
    c = np.zeros((n, 0))
    
    # Use normal constructor - it will handle empty inputs correctly
    pZ = PolyZonotope(c)
    
    return pZ 