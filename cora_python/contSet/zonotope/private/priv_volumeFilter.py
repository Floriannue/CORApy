"""
priv_volumeFilter - filters out generators by directly choosing the 
    smallest volume

Syntax:
    Gfinal = priv_volumeFilter(G, Z, varargin)

Inputs:
    G - cells of generator matrices
    Z - original zonotope
    nrOfPicks - number of parallelotopes that are picked

Outputs:
    Gfinal - final generator matrix

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2008 (MATLAB)
Last update:   15-September-2008 (MATLAB)
                14-March-2019 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import List, Optional
from ..zonotope import Zonotope
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues


def priv_volumeFilter(G: List[np.ndarray], Z: Zonotope, nrOfPicks: Optional[int] = None) -> List[np.ndarray]:
    """
    Private volume filter - filters out generators by directly choosing the 
    smallest volume
    
    Args:
        G: cells of generator matrices
        Z: original zonotope
        nrOfPicks: number of parallelotopes that are picked (default: 1)
        
    Returns:
        final generator matrix
    """
    # Parse input arguments
    if nrOfPicks is None:
        nrOfPicks = 1
    
    # Obtain dimension
    d = G[0].shape[0]
    
    # Init volume
    vol = np.zeros(len(G))
    
    # Determine generators by exact volume minimization:
    for i in range(len(G)):
        # Get transformation matrix P
        P = G[i]
        
        # Check rank of P
        if np.linalg.matrix_rank(P) < d:
            vol[i] = np.inf
        else:
            try:
                # Compute reduced zonotope
                P_inv = np.linalg.pinv(P)
                Ztrans = P_inv @ Z
                Zinterval = Ztrans.interval()
                Zred = P @ Zonotope(Zinterval.inf, Zinterval.sup - Zinterval.inf)
                
                # Compute volume (use volume_ function as in Python implementation)
                from ..volume_ import volume_
                vol[i] = volume_(Zred, 'exact')
            except:
                vol[i] = np.inf
    
    # Obtain indices corresponding to the smallest values (equivalent to MATLAB's mink)
    index = np.argsort(vol)[:nrOfPicks]
    
    Gfinal = []
    for i in range(nrOfPicks):
        Gfinal.append(G[index[i]])
    
    return Gfinal 