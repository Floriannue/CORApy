"""
matZonotope method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def matZonotope(Z: Zonotope):
    """
    Converts the given zonotope to a matrix zonotope
    
    Args:
        Z: zonotope object
        
    Returns:
        MatZonotope object
        
    Example:
        Z = Zonotope(np.random.randn(2, 1), np.random.randn(2, 10))
        matZ = matZonotope(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Extract zonotope properties
    c = Z.c
    G = Z.G
    
    # Extend generators
    n, h = G.shape
    G = G.reshape(n, 1, h)
    
    # Construct matrix zonotope
    # Note: MatZonotope class may not be implemented yet
    # This is a placeholder implementation
    matZ = {'type': 'matZonotope', 'center': c, 'generators': G}
    
    return matZ 