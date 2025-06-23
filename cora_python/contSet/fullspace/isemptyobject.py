"""
isemptyobject - checks whether a fullspace object contains any information

Syntax:
    res = isemptyobject(fs)

Inputs:
    fs - fullspace object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       07-June-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fullspace import Fullspace


def isemptyobject(fs: 'Fullspace') -> bool:
    """
    Checks whether a fullspace object contains any information
    
    Args:
        fs: fullspace object
        
    Returns:
        res: true if the fullspace is empty, false otherwise
    """
    
    # Fullspace is empty if dimension is 0 or not set
    return not hasattr(fs, 'n') or fs.n == 0 