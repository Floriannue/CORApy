"""
reduce - reduces the order of a zonotope, the resulting zonotope is an
    over-approximation of the original zonotope

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 24-January-2007 (MATLAB)
Last update: 15-September-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def reduce(Z: 'Zonotope', method: str, order: Optional[int] = 1, 
          filterLength: Optional[int] = None, option: Optional[str] = None) -> 'Zonotope':
    """
    Reduces the order of a zonotope
    
    Args:
        Z: Zonotope object
        method: Reduction method ('girard', 'combastel', 'pca', etc.)
        order: Order of reduced zonotope (default: 1)
        filterLength: Optional filter length for some methods
        option: Optional additional option
        
    Returns:
        Zonotope: Reduced zonotope object
        
    Raises:
        CORAError: If method is not supported
        
    Example:
        >>> Z = Zonotope([1, -1], [[2, 3, 1], [2, 3, -2]])
        >>> Z_red = reduce(Z, 'girard', 2)
    """
    from .zonotope import Zonotope
    
    # Handle default values
    if order is None:
        order = 1
    
    # Remove substring necessary for special reduction for polyZonotopes
    if method.startswith('approxdep_'):
        method = method.replace('approxdep_', '')
    
    # Select method
    if method == 'girard':
        return _reduce_girard(Z, order)
    elif method == 'combastel':
        return _reduce_combastel(Z, order)
    elif method == 'pca':
        return _reduce_pca(Z, order)
    elif method == 'methA':
        return _reduce_methA(Z, order)
    elif method == 'methB':
        return _reduce_methB(Z, order, filterLength)
    elif method == 'methC':
        return _reduce_methC(Z, order, filterLength)
    else:
        # For now, only implement the most common methods
        raise CORAError('CORA:wrongValue', 
                       f"Reduction method '{method}' not yet implemented. "
                       f"Available methods: 'girard', 'combastel', 'pca', 'methA', 'methB', 'methC'")


def _reduce_girard(Z: 'Zonotope', order: int) -> 'Zonotope':
    """
    Girard's method for zonotope order reduction (Sec. 4 in [2])
    
    This is the most commonly used reduction method.
    """
    from .zonotope import Zonotope
    
    # Get dimension and number of generators
    n = Z.dim()
    G = Z.G
    
    # If zonotope is already lower order or empty, return as-is
    if G.shape[1] <= order or G.shape[1] == 0:
        return Z.copy()
    
    # Compute norms of generators
    h = np.linalg.norm(G, axis=0, ord=1)  # L1 norm of each generator
    
    # Sort generators by norm (ascending order)
    idx = np.argsort(h)
    
    # Keep the largest generators
    keep_idx = idx[-(order):] if order > 0 else []
    reduce_idx = idx[:-(order)] if order > 0 else idx
    
    # Build reduced generator matrix
    if len(keep_idx) > 0:
        G_keep = G[:, keep_idx]
    else:
        G_keep = np.zeros((n, 0))
    
    # Sum up the reduced generators into interval hull
    if len(reduce_idx) > 0:
        G_reduce = G[:, reduce_idx]
        # Create interval hull: sum of absolute values of reduced generators
        d = np.sum(np.abs(G_reduce), axis=1, keepdims=True)
        # Convert to diagonal matrix representation
        G_interval = np.diag(d.flatten())
        # Combine with kept generators
        if G_keep.size > 0:
            G_new = np.hstack([G_keep, G_interval])
        else:
            G_new = G_interval
    else:
        G_new = G_keep
    
    return Zonotope(Z.c, G_new)


def _reduce_combastel(Z: 'Zonotope', order: int) -> 'Zonotope':
    """
    Combastel's method for zonotope order reduction (Sec. 3.2 in [4])
    
    This is a simplified implementation for now.
    """
    # For now, fall back to Girard's method
    return _reduce_girard(Z, order)


def _reduce_pca(Z: 'Zonotope', order: int) -> 'Zonotope':
    """
    PCA-based method for zonotope order reduction (Sec. III.A in [3])
    
    This is a simplified implementation for now.
    """
    # For now, fall back to Girard's method
    return _reduce_girard(Z, order)


def _reduce_methA(Z: 'Zonotope', order: int) -> 'Zonotope':
    """
    Method A for zonotope order reduction (Sec. 2.5.5 in [1])
    
    This is a simplified implementation for now.
    """
    # For now, fall back to Girard's method
    return _reduce_girard(Z, order)


def _reduce_methB(Z: 'Zonotope', order: int, filterLength: Optional[int] = None) -> 'Zonotope':
    """
    Method B for zonotope order reduction (Sec. 2.5.5 in [1])
    
    This is a simplified implementation for now.
    """
    # For now, fall back to Girard's method
    return _reduce_girard(Z, order)


def _reduce_methC(Z: 'Zonotope', order: int, filterLength: Optional[int] = None) -> 'Zonotope':
    """
    Method C for zonotope order reduction (Sec. 2.5.5 in [1])
    
    This is a simplified implementation for now.
    """
    # For now, fall back to Girard's method
    return _reduce_girard(Z, order) 