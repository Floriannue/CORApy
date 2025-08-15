"""
dim - returns the dimension of the ambient space of a constrained zonotope

Syntax:
    n = dim(cZ)

Inputs:
    cZ - conZonotope object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conZonotope import ConZonotope

def dim(cZ: 'ConZonotope') -> int:
    """
    Returns the dimension of the ambient space of the constrained zonotope.

    Args:
        cZ: ConZonotope object.

    Returns:
        Dimension of the ambient space.
    """
    if cZ._dim_val is not None:
        return cZ._dim_val
    # Fallback if _dim_val is not set (should not happen with proper construction)
    if cZ.c.size == 0:
        return 0
    return cZ.c.shape[0] 