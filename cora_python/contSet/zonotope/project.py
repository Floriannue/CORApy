"""
project - projects a zonotope onto the specified dimensions

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 15-September-2008 (MATLAB)
Last update: 20-October-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .zonotope import Zonotope

def project(Z: Zonotope, dims) -> Zonotope:
    """
    Projects a zonotope onto the specified dimensions
    
    Args:
        Z: Zonotope object
        dims: dimensions for projection (list or array of indices)
        
    Returns:
        Zonotope: projected zonotope
        
    Example:
        >>> Z = Zonotope([1, 0, 1], [[1, -1, 0], [0, 0, -1], [1, 0, 1]])
        >>> Z_proj = project(Z, [0, 2])  # Project onto 1st and 3rd dimensions
    """
    
    # Convert dims to numpy array for indexing
    dims = np.array(dims)
    
    # Project center and generators
    c_proj = Z.c[dims]
    
    if Z.G.shape[0] > 0:
        G_proj = Z.G[dims, :]
    else:
        G_proj = Z.G
    
    return Zonotope(c_proj, G_proj) 