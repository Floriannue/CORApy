"""
empty - instantiates an empty fullspace

Syntax:
    fs = empty(n)

Inputs:
    n - dimension

Outputs:
    fs - empty fullspace object

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
    from .fullspace import Fullspace


def empty(n: int = 0) -> 'Fullspace':
    """
    Instantiates an empty fullspace
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        fs: empty fullspace object
    """
    
    from .fullspace import Fullspace
    
    return Fullspace(n) 