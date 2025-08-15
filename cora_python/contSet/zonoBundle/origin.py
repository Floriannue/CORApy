"""
origin - instantiates a zonotope bundle representing the origin in R^n

Syntax:
    zB = origin(n)

Inputs:
    n - dimension

Outputs:
    zB - zonoBundle object representing the origin

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       21-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zonoBundle import ZonoBundle


def origin(n: int) -> 'ZonoBundle':
    """
    Instantiates a zonotope bundle representing the origin in R^n
    
    Args:
        n: dimension
        
    Returns:
        zB: zonoBundle object representing the origin
    """
    
    from .zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Create origin zonotope and put it in a bundle
    if n == 0:
        # In 0D, return empty bundle
        return ZonoBundle([])
    origin_zono = Zonotope.origin(n)
    return ZonoBundle([origin_zono])