"""
simplex method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def simplex(Z: Zonotope):
    """
    Enclose a zonotope by a simplex
    
    Args:
        Z: zonotope object
        
    Returns:
        Polytope object representing the simplex
        
    Example:
        Z = Zonotope(np.array([[1], [0]]), np.array([[1, -1, 0.5], [0, 1, 1]]))
        P = simplex(Z)
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Get dimension
    n = Z.c.shape[0]
    
    # Construct an n-dimensional standard simplex with origin 0
    V = np.eye(n + 1)
    B = _gram_schmidt(np.ones((n + 1, 1)))
    
    # Create polytope from the simplex
    from cora_python.contSet.polytope.polytope import Polytope
    P = Polytope((B[:, 1:].T @ V))
    
    # Compute the halfspace representation
    P.constraints()
    
    # Scale the simplex so that it tightly encloses the zonotope
    A = P.A
    if A is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Polytope constraints are None')
    
    # Compute supremum of interval(A*Z)
    from cora_python.contSet.interval.interval import interval
    Z_interval = interval(Z)
    if Z_interval is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Could not create interval from zonotope')
    
    b = Z_interval.supremum()
    if b is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Interval supremum is None')
    
    # Create final polytope
    P = Polytope(A, b)
    
    return P


def _gram_schmidt(v: np.ndarray) -> np.ndarray:
    """
    Gram-Schmidt orthogonalization
    
    Args:
        v: input vector
        
    Returns:
        Orthogonalized matrix
    """
    n = v.shape[0]
    B = np.zeros((n, n))
    B[:, 0] = v.flatten()
    
    for i in range(1, n):
        B[:, i] = np.eye(n)[:, i]
        for j in range(i):
            B[:, i] = B[:, i] - np.dot(B[:, i], B[:, j]) / np.dot(B[:, j], B[:, j]) * B[:, j]
        # Normalize
        norm = np.linalg.norm(B[:, i])
        if norm > 0:
            B[:, i] = B[:, i] / norm
    
    return B 