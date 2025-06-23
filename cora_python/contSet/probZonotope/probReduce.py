"""
probReduce - Reduces the number of single Gaussian distributions to
   the dimension of the system

Syntax:
   probZ = probReduce(probZ)

Inputs:
   probZ - probabilistic zonotope object

Outputs:
   probZ - probabilistic zonotope object

Example:
   Z1 = [10 1 -2; 0 1 1]
   Z2 = [0.6 1.2; 0.6 -1.2]
   probZ = probZonotope(Z1,Z2)
   probZred = probReduce(probZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       27-August-2007 (MATLAB)
Last update:   26-February-2008 (MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .probZonotope import ProbZonotope


def probReduce(probZ: 'ProbZonotope') -> 'ProbZonotope':
    """
    Reduces the number of single Gaussian distributions to the dimension of the system
    
    Args:
        probZ: probabilistic zonotope object
        
    Returns:
        probZ: probabilistic zonotope object
    """
    
    if probZ.gauss != 1:
        # get new sigma matrix
        G = probZ.g
        newSigma = G @ G.T
    else:
        newSigma = probZ.cov
    
    # ensure symmetry for numerical stability
    newSigma = 0.5 * (newSigma + newSigma.T)
    
    # get eigenvalue, eigenvectors of newSigma
    W, V = np.linalg.eig(newSigma)
    
    # compute new generators
    probZ.g = V @ np.sqrt(np.diag(W))
    
    return probZ 