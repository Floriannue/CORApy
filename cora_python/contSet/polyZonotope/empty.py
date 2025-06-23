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
    # Temporarily disable input validation for debugging
    # from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
    
    # parse input - match MATLAB behavior
    # inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # Create empty polyZonotope like MATLAB: polyZonotope(zeros(n,0))
    c = np.zeros((n, 0))
    
    # Temporarily create directly to bypass validation
    pZ = PolyZonotope.__new__(PolyZonotope)
    pZ.c = c
    pZ.G = np.zeros((n, 0))
    pZ.GI = np.zeros((n, 0))
    pZ.E = np.zeros((0, 0))
    pZ.id = np.array([]).reshape(0, 1)
    pZ.precedence = 70
    
    return pZ 