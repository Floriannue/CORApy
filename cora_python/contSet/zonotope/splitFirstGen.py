"""
splitFirstGen method for zonotope class
"""

import numpy as np
from typing import List
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def splitFirstGen(Z: List[Zonotope]) -> List[Zonotope]:
    """
    Splits first generator, which is in direction of the vector field
    
    Args:
        Z: list of zonotope objects
        
    Returns:
        List of remaining zonotope objects
        
    Example:
        Z = [Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))]
        Znew = splitFirstGen(Z)
    """
    # Initialize Znew
    Znew = []
    
    # Split first generator
    for i in range(len(Z)):
        # Find longest generator
        G = Z[i].G
        if G is None or G.size == 0:
            continue
            
        h = []
        for j in range(G.shape[1]):
            h.append(np.linalg.norm(G[:, j:j+1].T @ G, ord=1))
        
        # Find index of longest generator
        index = np.argmax(h)
        
        # Split longest generator
        from .split import split
        Ztemp = split(Z[i], index)
        
        # Write to Znew
        Znew.extend(Ztemp)
    
    return Znew 