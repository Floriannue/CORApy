"""
project - projects a zonotope onto the specified dimensions

Syntax:
    Z = project(Z,dims)

Inputs:
    Z - (zonotope) zonotope
    dims - dimensions for projection

Outputs:
    Z - (zonotope) projected zonotope

Example: 
    Z = Zonotope(np.array([[1, -1, 0], [0, 0, -1], [1, 0, 1]]))
    Z = project(Z, [0, 2])

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 15-September-2008 (MATLAB)
Last update: 20-October-2023 (TL, correct projection for G\in\R^{n x 0}) (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
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
    
    # Project center (matching MATLAB: Z.c = Z.c(dims,:))
    c_proj = Z.c[dims]
    
    # Project generators (matching MATLAB: if size(Z.G,1) > 0; Z.G = Z.G(dims,:); end)
    if Z.G.shape[0] > 0:
        G_proj = Z.G[dims, :]
    else:
        G_proj = Z.G
    
    return Zonotope(c_proj, G_proj) 