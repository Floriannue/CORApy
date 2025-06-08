"""
isemptyobject - checks whether a zonotope contains any information at
    all; consequently, the set is interpreted as the empty set

Syntax:
    res = isemptyobject(Z)

Inputs:
    Z - zonotope object

Outputs:
    res - true/false

Example:
    Z = zonotope([2, 1])
    isemptyobject(Z)  # False

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       24-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np


def isemptyobject(Z):
    """
    Checks whether a zonotope contains any information at all
    
    Args:
        Z: zonotope object
        
    Returns:
        bool: True if zonotope is empty, False otherwise
    """
    # A zonotope is empty if it has no center or the center is empty
    return Z.c is None or Z.c.size == 0 