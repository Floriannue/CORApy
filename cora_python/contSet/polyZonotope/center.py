"""
center - returns the center of a polynomial zonotope

Syntax:
    c = center(pZ)

Inputs:
    pZ - polyZonotope object

Outputs:
    c - center vector
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def center(pZ: 'PolyZonotope') -> np.ndarray:
    return pZ.c.copy()
