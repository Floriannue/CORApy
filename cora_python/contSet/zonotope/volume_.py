"""
volume_ - computes the volume of a zonotope

Authors: Matthias Althoff (MATLAB)
         Automatic python translation: Florian Nüssel BA 2025
Written: 24-August-2007 (MATLAB)
Last update: 19-July-2010 (MATLAB)
            02-September-2019 (incl. approx) (MATLAB)
            04-May-2020 (MW, add vol=0 cases) (MATLAB)
            09-September-2020 (MA, Alamo approx added, reduce changed) (MATLAB)
            27-July-2022 (ME, included batchCombinator) (MATLAB)
            18-August-2022 (MW, include standardized preprocessing) (MATLAB)
Last revision: 27-March-2023 (MW, rename volume_) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional
from itertools import combinations
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def volume_(Z: Zonotope, method: str = 'exact', order: Optional[int] = None) -> float:
    """
    Computes the volume of a zonotope
    
    Args:
        Z: zonotope object
        method: method for approximation 
               - 'exact' (default)
               - 'reduce' for reduced zonotope with order o
               - 'alamo', see [2]
        order: zonotope order for reduction before computation
        
    Returns:
        Volume of the zonotope
        
    References:
        [1] E. Grover et al. "Determinants and the volumes of parallelotopes 
            and zonotopes", 2010 
        [2] Alamo et al. "Bounded error identification of systems with 
            time-varying parameters", TAC 2006.
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