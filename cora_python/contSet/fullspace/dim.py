"""
dim - returns the dimension of the ambient space of the full space

Syntax:
    n = dim(fs)

Inputs:
    fs - fullspace object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       15-September-2019 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fullspace import Fullspace


def dim(fs: 'Fullspace') -> int:
    """
    Returns the dimension of the ambient space of the full space
    
    Args:
        fs: fullspace object
        
    Returns:
        n: dimension of the ambient space
    """
    
    return fs.dimension 