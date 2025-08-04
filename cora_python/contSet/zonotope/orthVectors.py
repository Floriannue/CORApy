"""
orthVectors - computes remaining orthogonal vectors when the zonotope is
    not full dimensional

Syntax:
    V = orthVectors(Z)

Inputs:
    Z - zonotope

Outputs:
    V - orthogonal vectors in matrix form

Example:

Other m-files required: vertices, polytope
Subfunctions: none
MAT-files required: none

See also: interval, vertices

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       17-January-2012 (MATLAB)
Last update:   27-August-2019 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from .zonotope import Zonotope
from cora_python.g.functions.helper.sets.contSet.zonotope.ndimCross import ndimCross


def orthVectors(Z: Zonotope) -> np.ndarray:
    """
    Computes remaining orthogonal vectors when the zonotope is not full dimensional
    
    Args:
        Z: zonotope object
        
    Returns:
        Orthogonal vectors in matrix form
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1], [0]]))
        V = orthVectors(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise ValueError("Zonotope center or generators are None")
    
    # Determine missing vectors
    n = Z.c.shape[0]
    G = Z.G
    nrGens = G.shape[1]
    nrOfVectors = n - nrGens
    
    # Compute missing vectors
    if nrOfVectors > 0:
        # Obtain set of random values
        if nrOfVectors > 1:
            randMat = np.random.rand(n, nrOfVectors - 1)
        else:
            randMat = np.empty((n, 0))
        
        for iVec in range(nrOfVectors):
            basis = np.hstack([G, randMat])
            gNew = ndimCross(basis)
            gNew = gNew / np.linalg.norm(gNew)
            
            # Update G, randMat
            G = np.hstack([G, gNew])
            if randMat.size > 0:
                randMat = randMat[:, 1:]
        
        V = G[:, nrGens:n]
    else:
        V = np.empty((n, 0))
    
    return V 