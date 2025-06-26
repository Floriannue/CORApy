"""
generatorLength method for zonotope class
"""

import numpy as np
from .zonotope import Zonotope


def generatorLength(Z: Zonotope) -> np.ndarray:
    """
    Returns the lengths of the generators
    
    Args:
        Z: zonotope object
        
    Returns:
        Vector of generator lengths
    """
    return np.linalg.norm(Z.G, axis=0) 