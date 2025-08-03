"""
underapproximate method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def underapproximate(Z: Zonotope, S: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns the vertices of an underapproximation. The underapproximation is computed 
    by finding the vertices that are extreme in the direction of a set of vectors, 
    stored in the matrix S. If S is not specified, it is constructed by the vectors 
    spanning an over-approximative parallelotope.
    
    Args:
        Z: zonotope object
        S: matrix of direction vectors (optional)
        
    Returns:
        Vertices of the underapproximation
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        V = underapproximate(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Extract center and generators
    c = Z.c
    G = Z.G
    
    if S is None:
        rows, cols = G.shape
        if rows >= cols:
            S = G
        else:
            # Use dominantDirections for proper direction selection
            from .dominantDirections import dominantDirections
            try:
                S = dominantDirections(Z)
            except:
                # Fallback: use the first n generators as directions
                n = rows
                S = G[:, :n]
    
    # Handle case where S is empty (e.g., zero generators)
    if S.size == 0:
        return np.empty((rows, 0))
    
    # Handle case where all generators are zero (after filtering)
    if np.all(G == 0):
        return np.empty((rows, 0))
    
    # Obtain extreme vertices along directions in S
    V = np.zeros((S.shape[0], 2 * S.shape[1]))
    
    for i in range(S.shape[1]):
        posVertex = c.copy()
        negVertex = c.copy()
        
        for iGen in range(G.shape[1]):
            s = np.sign(S[:, i].T @ G[:, iGen])
            # Ensure proper vector addition by reshaping
            posVertex = posVertex + s * G[:, iGen].reshape(-1, 1)
            negVertex = negVertex - s * G[:, iGen].reshape(-1, 1)
        
        # MATLAB uses 1-based indexing: V(:,2*i-1) and V(:,2*i)
        # Python uses 0-based indexing: V[:,2*i] and V[:,2*i+1]
        # Ensure vertices are column vectors
        V[:, 2*i] = posVertex.flatten()
        V[:, 2*i + 1] = negVertex.flatten()
    
    return V 