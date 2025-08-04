"""
full - Converts a zonotope (center and generator) into full
         representation (from sparse)

Syntax:
    Z_full = full(Z)

Inputs:
    Z - zonotope object (assumed to be sparse)

Outputs:
    Z_full - zonotope with full center vector and generator matrix


Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Maximilian Perschl (MATLAB)
               Python translation by AI Assistant
Written:       02-June-2025 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def full(Z: Zonotope) -> Zonotope:
    """
    Converts a zonotope (center and generator) into full representation (from sparse)
    
    Args:
        Z: zonotope object (assumed to be sparse)
        
    Returns:
        Zonotope with full center vector and generator matrix
        
    Example:
        c = np.random.rand(10, 1)
        c[c < 0.8] = 0
        G = np.random.rand(10, 10)
        G[G < 0.8] = 0
        Z = Zonotope(sp.csr_matrix(c), sp.csr_matrix(G))
        Z_full = full(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Convert to full representation
    if sp.issparse(Z.c):
        c_full = Z.c.toarray()
    else:
        c_full = Z.c
    
    if sp.issparse(Z.G):
        G_full = Z.G.toarray()
    else:
        G_full = Z.G
    
    # Create new zonotope with full matrices
    Z_full = Zonotope(c_full, G_full)
    
    return Z_full 