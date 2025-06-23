"""
origin - instantiates a fullspace representing the origin in R^n (which is R^n itself)

Syntax:
    fs = origin(n)

Inputs:
    n - dimension

Outputs:
    fs - fullspace object representing R^n

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
    from .fullspace import Fullspace


def origin(n: int) -> 'Fullspace':
    """
    Instantiates a fullspace representing R^n
    
    Args:
        n: dimension
        
    Returns:
        fs: fullspace object representing R^n
    """
    
    from .fullspace import Fullspace
    
    return Fullspace(n) 