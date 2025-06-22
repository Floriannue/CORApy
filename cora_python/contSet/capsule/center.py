"""
center - Returns the center of a capsule

Syntax:
    c = center(C)

Inputs:
    C - capsule

Outputs:
    c - center of the capsule C

Example: 
    C = Capsule([1, 1, 0], [0.5, -1, 1], 0.5)
    c = center(C)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       05-March-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capsule import Capsule


def center(C: 'Capsule') -> np.ndarray:
    """
    Returns the center of a capsule
    
    Args:
        C: capsule object
        
    Returns:
        c: center of the capsule C
    """
    return C.c 