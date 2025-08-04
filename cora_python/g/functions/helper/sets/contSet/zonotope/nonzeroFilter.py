"""
nonzeroFilter - filters out generators of length 0

Syntax:
    G = nonzeroFilter(G)
    G = nonzeroFilter(G,tol)

Inputs:
    G - matrix of generators
    tol - (optional) tolerance

Outputs:
    G - reduced matrix of generators

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: 

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2008 (MATLAB)
Last update:   02-September-2019 (MATLAB)
               24-January-2024 (MW, add tolerance, MATLAB)
Python translation: 2025
"""

import numpy as np


def nonzeroFilter(G: np.ndarray, tol: float = None) -> np.ndarray:
    """
    Filters out generators of length 0
    
    Args:
        G: matrix of generators
        tol: tolerance (optional)
        
    Returns:
        reduced matrix of generators
    """
    # Delete zero-generators (any non-zero entry in a column)
    G_filtered = G[:, np.any(G != 0, axis=0)]
    
    if tol is not None:
        # Also remove generators with norm below tolerance
        G_norms = np.linalg.norm(G_filtered, axis=0)
        G_filtered = G_filtered[:, G_norms > tol]
    
    return G_filtered 