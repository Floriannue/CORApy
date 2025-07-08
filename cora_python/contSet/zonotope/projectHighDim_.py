"""
projectHighDim_ method for zonotope class
"""

import numpy as np
from typing import List
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def projectHighDim_(Z: Zonotope, N: int, proj: List[int]) -> Zonotope:
    """
    Project a zonotope to a higher-dimensional space
    
    Args:
        Z: zonotope object
        N: dimension of the higher dimensional space
        proj: states of the high dimensional space that correspond to the
              states of the low dimensional zonotope object
              
    Returns:
        Zonotope object in the higher-dimensional space
        
    Example:
        Z = Zonotope(np.array([[-1], [1]]), np.array([[3, 2, -1], [2, -1, 2]]))
        Z_ = projectHighDim_(Z, 5, [1, 3])
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Instantiate all-zero center/generator matrix in higher-dimensional space
    cnew = np.zeros((N, 1))
    Gnew = np.zeros((N, Z.G.shape[1]))
    
    # Insert input argument into desired dimensions of higher-dimensional space
    cnew[proj, :] = Z.c
    Gnew[proj, :] = Z.G
    
    # Instantiate zonotope object
    Z_projected = Zonotope(cnew, Gnew)
    
    return Z_projected 