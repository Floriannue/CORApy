"""
dominantDirections method for zonotope class
"""

import numpy as np
from typing import Optional, Tuple
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck


def dominantDirections(Z: Zonotope, filterLength1: Optional[int] = None, 
                      filterLength2: Optional[int] = None) -> np.ndarray:
    """
    Computes the directions that span a parallelotope which tightly encloses a zonotope Z
    
    Args:
        Z: zonotope object
        filterLength1: length of the length filter (default: n+5)
        filterLength2: length of the generator volume filter (default: n+3)
        
    Returns:
        Matrix containing the dominant directions as column vectors
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), -1 + 2*np.random.rand(2, 20))
        S = dominantDirections(Z)
    """
    # Get dimension
    if Z.c is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Center is None')
    n = Z.c.shape[0]
    
    # Parse input arguments
    defaults = [n+5, n+3]
    args = [filterLength1, filterLength2]
    result, _ = setDefaultValues(defaults, args)
    filterLength1, filterLength2 = result
    
    # Check input arguments
    inputArgsCheck([
        [Z, 'att', 'zonotope'],
        [filterLength1, 'att', 'numeric', ['nonnan', 'scalar', 'positive']],
        [filterLength2, 'att', 'numeric', ['nonnan', 'scalar', 'positive']]
    ])
    
    # Delete zero-generators
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Generator matrix is None')
    G = _nonzeroFilter(Z.G)
    
    # Number of generators
    nrOfGens = G.shape[1]
    
    # Correct filter length if necessary
    if filterLength1 is not None and filterLength1 > nrOfGens:
        filterLength1 = nrOfGens
    
    if filterLength2 is not None and filterLength2 > nrOfGens:
        filterLength2 = nrOfGens
    
    # Length filter
    if filterLength1 is None:
        filterLength1 = nrOfGens
    G = _lengthFilter(G, filterLength1)
    
    # Apply generator volume filter
    if filterLength2 is None:
        filterLength2 = nrOfGens
    Gcells = _generatorVolumeFilter(G, filterLength2)
    
    # Pick generator with the best volume
    G_picked = _volumeFilter(Gcells, Z)
    
    # Select dominant directions S
    S = G_picked[:, :n]
    
    return S


def _nonzeroFilter(G: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Filters out generators of length 0
    
    Args:
        G: matrix of generators
        tol: tolerance
        
    Returns:
        reduced matrix of generators
    """
    # Delete zero-generators (any non-zero entry in a column)
    G_filtered = G[:, np.any(G != 0, axis=0)]
    
    # Also remove generators with norm below tolerance
    G_norms = np.linalg.norm(G_filtered, axis=0)
    G_filtered = G_filtered[:, G_norms > tol]
    
    return G_filtered


def _lengthFilter(G: np.ndarray, filterLength: int) -> np.ndarray:
    """
    Length filter for generators
    
    Args:
        G: matrix of generators
        filterLength: number of generators to keep
        
    Returns:
        filtered matrix of generators
    """
    # Compute lengths of generators
    lengths = np.linalg.norm(G, axis=0)
    
    # Sort by length (descending) and keep top filterLength
    sorted_indices = np.argsort(lengths)[::-1]
    keep_indices = sorted_indices[:filterLength]
    
    return G[:, keep_indices]


def _generatorVolumeFilter(G: np.ndarray, filterLength: int) -> list:
    """
    Generator volume filter
    
    Args:
        G: matrix of generators
        filterLength: number of generators to keep
        
    Returns:
        list of generator combinations
    """
    n = G.shape[0]
    nrOfGens = G.shape[1]
    
    # If we have fewer generators than dimensions, return all
    if nrOfGens <= n:
        return [G]
    
    # Create combinations of n generators
    from itertools import combinations
    
    Gcells = []
    for combo in combinations(range(nrOfGens), n):
        Gcells.append(G[:, list(combo)])
    
    # Limit to filterLength combinations
    if len(Gcells) > filterLength:
        Gcells = Gcells[:filterLength]
    
    return Gcells


def _volumeFilter(Gcells: list, Z: Zonotope) -> np.ndarray:
    """
    Volume filter to pick the best generator combination
    
    Args:
        Gcells: list of generator combinations
        Z: original zonotope
        
    Returns:
        best generator matrix
    """
    best_volume = -1
    best_G = None
    
    for G_combo in Gcells:
        # Create temporary zonotope with this combination
        temp_Z = Zonotope(Z.c, G_combo)
        
        # Compute volume using the volume_ function
        try:
            from .volume_ import volume_
            volume = volume_(temp_Z, 'exact')
            if volume > best_volume:
                best_volume = volume
                best_G = G_combo
        except:
            # If volume computation fails, skip this combination
            continue
    
    # If no valid combination found, return the first one
    if best_G is None and Gcells:
        best_G = Gcells[0]
    elif best_G is None:
        # If no combinations available, return empty array
        if Z.c is None:
            return np.empty((0, 0))
        return np.empty((Z.c.shape[0], 0))
    
    return best_G 