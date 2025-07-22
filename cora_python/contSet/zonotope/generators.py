"""
generators method for zonotope class (legacy/functional access only)

WARNING: Do NOT import this function in any file that uses Zonotope.generators property.
Importing this function will shadow the property and break instance behavior.
Use only for legacy or functional access, not in new code.
"""

import numpy as np
from .zonotope import Zonotope

def generators(Z: Zonotope) -> np.ndarray:
    """
    Returns the generator matrix of a zonotope
    Args:
        Z: zonotope object
    Returns:
        Generator matrix
    """
    return Z.G 