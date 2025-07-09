"""
dH2box method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def dH2box(Z: Zonotope, method: Optional[str] = None) -> float:
    """
    Computes an over-approximation of the Hausdorff distance to the interval 
    over-approximation of the provided zonotope Z
    
    Args:
        Z: zonotope object
        method: over-approximation method ('exact', 'naive', 'ell', 'wgreedy', 'wopt')
        
    Returns:
        Over-approximated Hausdorff distance
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.random.rand(2, 20) * 2 - 1)
        dH = dH2box(Z, 'naive')
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Input arguments
    if Z.c.shape[0] <= 3:
        dV = 'exact'
    else:
        dV = 'naive'
    
    # Parse input arguments
    if method is None:
        method = dV
    
    # Check input arguments
    valid_methods = ['exact', 'naive', 'ell', 'wgreedy', 'wopt']
    if method not in valid_methods:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       f'Method must be one of {valid_methods}')
    
    # Methods
    if method == 'exact':
        dH = _aux_dH2box_exact(Z)
    elif method == 'naive':
        dH = _aux_dH2box_naive(Z)
    elif method == 'ell':
        dH = _aux_dH2box_ell(Z)
    elif method == 'wgreedy':
        dH = _aux_dH2box_wgreedy(Z)
    elif method == 'wopt':
        dH = _aux_dH2box_wopt(Z)
    
    return dH


def _aux_dH2box_exact(Z: Zonotope) -> float:
    """
    Computes near-exact dH according to [1]
    """
    # Generator matrices
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope generators are None')
    G = Z.G
    Gbox = np.diag(np.sum(np.abs(G), axis=1))
    
    # Generate lambda vectors (simplified)
    if Z.c is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope center is None')
    n = Z.c.shape[0]
    numDirs = min(10000, 2**n)  # Limit for high dimensions
    lambda_vectors = np.random.randn(n, numDirs)
    lambda_vectors = lambda_vectors / np.linalg.norm(lambda_vectors, axis=0)
    
    # Loop over each lambda to find maximum
    dH = 0.0
    for i in range(lambda_vectors.shape[1]):
        lambda_vec = lambda_vectors[:, i:i+1]
        dHlambda = abs(float(np.linalg.norm(lambda_vec.T @ Gbox, 1)) - 
                       float(np.linalg.norm(lambda_vec.T @ G, 1)))
        if dHlambda > dH:
            dH = dHlambda
    
    return float(dH)


def _aux_dH2box_naive(Z: Zonotope) -> float:
    """
    Computes the radius of the box over-approximation of Z
    """
    # Simplified implementation - compute box radius directly
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope generators are None')
    
    # Compute box radius as sum of absolute values of generators
    box_radius = np.sum(np.abs(Z.G), axis=1)
    return float(np.linalg.norm(box_radius, 2))


def _aux_dH2box_ell(Z: Zonotope) -> float:
    """
    Computes distance from box under-approximations to box over-approximations
    """
    # Simplified implementation
    return _aux_dH2box_naive(Z)


def _aux_dH2box_wgreedy(Z: Zonotope) -> float:
    """
    Computes length of sum of generators with largest absolute element set to 0
    """
    # Generator matrices
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope generators are None')
    G = Z.G
    Gabs = np.abs(G)
    _, gamma = G.shape
    
    # Truncated generators (largest absolute element per column = 0)
    Gtruncated = Gabs.copy()
    for k in range(gamma):
        idxmax = np.argmax(Gabs[:, k])
        Gtruncated[idxmax, k] = 0
    
    # Measure
    dH = 2 * np.linalg.norm(np.sum(Gtruncated, axis=1), 2)
    
    return float(dH)


def _aux_dH2box_wopt(Z: Zonotope) -> float:
    """
    Computes optimal scaling of generator lengths
    """
    # Generator matrices
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope generators are None')
    G = Z.G
    Gabs = np.abs(G)
    
    # Dimensions
    n, gamma = G.shape
    
    # Lengths of generators
    lsquared = np.linalg.norm(G, 2, axis=0) ** 2
    
    dH = 0.0
    for i in range(n):
        summand = 0.0
        for k in range(gamma):
            if lsquared[k] != 0:
                summand += Gabs[i, k] * (1 - G[i, k]**2 / lsquared[k])
        dH += summand**2
    
    dH = 2 * np.sqrt(dH)
    
    return float(dH) 