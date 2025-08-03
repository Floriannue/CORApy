"""
volume_ - computes the volume of a zonotope

Syntax:
    vol = volume_(Z, method, order)

Inputs:
    Z - zonotope object
    method - (optional) method for approximation 
           - 'exact' (default)
           - 'reduce' for reduced zonotope with order o
           - 'alamo', see [2]
    order - (optional) zonotope order for reduction before computation

Outputs:
    vol - volume

Example:
    from cora_python.contSet.zonotope import Zonotope, volume_
    import numpy as np
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, -1, 0], [0, 0, -1]]))
    vol = volume_(Z)

References:
    [1] E. Grover et al. "Determinants and the volumes of parallelotopes and zonotopes", 2010 
    [2] Alamo et al. "Bounded error identification of systems with time-varying parameters", TAC 2006.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/volume

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       24-August-2007 (MATLAB)
Last update:   18-August-2022 (MW, include standardized preprocessing) (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional
from itertools import combinations
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def volume_(Z: Zonotope, method: str = 'exact', order: Optional[int] = None) -> float:
    """
    Computes the volume of a zonotope.
    """
    # Get dimension and number of generators
    G = Z.G
    n, nrOfGens = G.shape
    
    # Non full-dimensional set
    if nrOfGens < n or np.linalg.matrix_rank(G) < n:
        return 0.0
    
    # Exact computation
    if method == 'exact':
        # Exact calculation using all possible parallelotopes
        
        accVol = 0.0
        
        # Return all possible parallelotopes which together form the zonotope
        # Use combinations to get all n-element subsets of generators
        for combo in combinations(range(nrOfGens), n):
            try:
                # Extract the n generators for this parallelotope
                G_combo = G[:, combo]
                # Add volume of this parallelotope
                currVol = abs(np.linalg.det(G_combo))
                accVol += currVol
            except Exception:
                raise CORAerror('CORA:specialError',
                              'Parallelotope volume could not be computed.')
        
        # Multiply result by factor 2^n
        vol = (2**n) * accVol
        
    # Over-approximative volume using order reduction
    elif method == 'reduce':
        if order is None:
            raise CORAerror('CORA:wrongValue', 
                           'Order parameter required for reduce method')
        # Reduce zonotope
        Zred = Z.reduce('pca', order)
        vol = Zred.volume_()
        
    # Approximation according to [2]    
    elif method == 'alamo':
        vol = (2**n) * np.sqrt(np.linalg.det(G @ G.T))
        
    else:
        raise CORAerror('CORA:wrongValue',
                       f"Unknown method: {method}. Must be 'exact', 'reduce', or 'alamo'")
    
    return vol 