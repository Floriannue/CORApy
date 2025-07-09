"""
splitLongestGen method for zonotope class
"""

import numpy as np
from typing import List
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def splitLongestGen(Z: Zonotope) -> List[Zonotope]:
    """
    Splits the longest generator
    
    Args:
        Z: zonotope object
        
    Returns:
        List storing the split zonotope objects
        
    Example:
        Z = Zonotope(np.array([[1], [0]]), np.array([[1, 3, -2, -1], [0, 2, -1, 1]]))
        Znew = splitLongestGen(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Object properties
    c = Z.c
    G = Z.G
    
    # Determine longest generator
    len_vec = np.sum(G**2, axis=0)
    ind = np.argmax(len_vec)
    
    # Split longest generator
    c1 = c + 0.5 * G[:, ind:ind+1]
    G1 = G.copy()
    G1[:, ind:ind+1] = 0.5 * G1[:, ind:ind+1]
    
    c2 = c - 0.5 * G[:, ind:ind+1]
    G2 = G.copy()
    G2[:, ind:ind+1] = 0.5 * G2[:, ind:ind+1]
    
    Znew = [Zonotope(c1, G1), Zonotope(c2, G2)]
    
    return Znew 