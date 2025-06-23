"""
sigma - returns Sigma matrix of a probabilistic zonotope

Syntax:
   sig = sigma(probZ)

Inputs:
   probZ - probabilistic zonotope object

Outputs:
   sig - sigma matrix

Example:
   Z1 = [10 1 -2; 0 1 1]
   Z2 = [0.6 1.2; 0.6 -1.2]
   probZ = probZonotope(Z1,Z2)
   sig = sigma(probZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       28-August-2007 (MATLAB)
Last update:   26-February-2008 (MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .probZonotope import ProbZonotope


def sigma(probZ: 'ProbZonotope') -> np.ndarray:
    """
    Returns Sigma matrix of a probabilistic zonotope
    
    Args:
        probZ: probabilistic zonotope object
        
    Returns:
        sig: sigma matrix
    """
    
    # reduce probabilistic zonotope first
    from .probReduce import probReduce
    probZ = probReduce(probZ)
    
    # get new sigma matrix
    G = probZ.g
    sig = G @ G.T
    
    return sig 