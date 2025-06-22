"""
dim - returns the dimension of the ambient space of a capsule

Syntax:
    n = dim(C)

Inputs:
    C - capsule object

Outputs:
    n - dimension of the ambient space

Example: 
    C = Capsule([1, 1, 0], [0.5, -1, 1], 0.5)
    n = dim(C)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       15-September-2019 (MATLAB)
Last update:   12-March-2021 (MW, add empty case, MATLAB)
               09-January-2024 (MW, simplify, MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capsule import Capsule


def dim(C: 'Capsule') -> int:
    """
    Returns the dimension of the ambient space of a capsule
    
    Args:
        C: capsule object
        
    Returns:
        n: dimension of the ambient space
    """
    return C.c.shape[0] 