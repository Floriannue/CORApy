"""
generators method for zonotope class
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