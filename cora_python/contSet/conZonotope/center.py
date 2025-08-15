"""
center - returns the center of a constrained zonotope

Syntax:
    c = center(cZ)
"""

import numpy as np
from .conZonotope import ConZonotope


def center(cZ: ConZonotope) -> np.ndarray:
    return cZ.c.reshape(-1, 1) if cZ.c is not None else np.array([]).reshape(0, 1)


