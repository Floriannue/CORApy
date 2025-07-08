"""
quadMap_parallel method for zonotope class
"""

import numpy as np
from typing import List
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def quadMap_parallel(Z: Zonotope, Q: List[np.ndarray]) -> Zonotope:
    """
    Computes {Q_{ijk}*x_j*x_k|x \in Z} using parallel processing
    
    Args:
        Z: zonotope object
        Q: quadratic coefficients as a list of matrices
        
    Returns:
        Zonotope object
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 1], [1, 0]]))
        Q = [np.array([[0.5, 0.5], [0, -0.5]]), 
             np.array([[-1, 0], [1, 1]])]
        res = quadMap_parallel(Z, Q)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Get center and generator matrix of zonotope
    Zmat = np.hstack([Z.c, Z.G])
    dimQ = len(Q)
    gens = Z.G.shape[1]
    
    # Initialize results
    c = np.zeros((dimQ, 1))
    G = [None] * dimQ
    
    # For each dimension, compute generator elements
    for i in range(dimQ):
        # Pure quadratic evaluation
        quadMat = Zmat.T @ Q[i] @ Zmat
        
        # Center
        ind = np.arange(gens)
        c[i, 0] = quadMat[0, 0] + 0.5 * np.sum(np.diag(quadMat[1:, 1:]))
        
        # Generators with center
        G[i] = np.zeros(2 * gens + gens * (gens - 1) // 2)
        G[i][:gens] = quadMat[0, 1:gens+1] + quadMat[1:gens+1, 0]
        
        # Generators from diagonal elements
        G[i][gens:2*gens] = 0.5 * np.diag(quadMat[1:gens+1, 1:gens+1])
        
        # Generators from other elements
        counter = 0
        for j in range(gens):
            kInd = np.arange(j+1, gens)
            if len(kInd) > 0:
                G[i][2*gens + counter:2*gens + counter + len(kInd)] = (
                    quadMat[j+1, kInd+1] + quadMat[kInd+1, j+1]
                )
                counter += len(kInd)
    
    # Convert G to matrix
    Gmat = np.array(G)
    
    # Generate new zonotope
    Z_result = Zonotope(c, Gmat.T)
    
    return Z_result 