"""
dim - returns the dimension of the ambient space of the constrained polynomial zonotope

Syntax:
   n = dim(cPZ)

Inputs:
   cPZ - conPolyZono object

Outputs:
   n - dimension of the ambient space

Authors:       Niklas Kochdumper (MATLAB)
Automatic python translation: Florian Nüssel BA 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conPolyZono import ConPolyZono

def dim(cPZ: 'ConPolyZono') -> int:
    """
    Returns the dimension of the ambient space of the constrained polynomial zonotope.

    Args:
        cPZ: ConPolyZono object.

    Returns:
        Dimension of the ambient space.
    """
    if cPZ._dim_val is not None:
        return cPZ._dim_val
    # Fallback if _dim_val is not set (should not happen with proper construction)
    if cPZ.c.size == 0:
        return 0
    return cPZ.c.shape[0]
