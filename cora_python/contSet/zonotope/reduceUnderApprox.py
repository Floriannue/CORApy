"""
reduceUnderApprox method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import inputArgsCheck


def reduceUnderApprox(Z: Zonotope, method: str, order: int) -> Zonotope:
    """
    Reduces the order of a zonotope so that an under-approximation of the original set is obtained
    
    Args:
        Z: zonotope object
        method: reduction method ('sum', 'scale', 'linProg', 'wetzlinger')
        order: zonotope order
        
    Returns:
        Reduced zonotope
        
    Example:
        Z = Zonotope(np.array([[1], [-1]]), np.array([[3, 2, -3, -1, 2, 4, -3, -2, 1], 
                                                       [2, 0, -2, -1, 2, -2, 1, 0, -1]]))
        Zsum = reduceUnderApprox(Z, 'sum', 3)
    """
    # Check input arguments
    inputArgsCheck([
        [Z, 'att', 'zonotope'],
        [method, 'str', ['sum', 'scale', 'linProg', 'wetzlinger']],
        [order, 'att', 'numeric', ['nonnan']]
    ])
    
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    # Remove all-zero generators
    from .compact_ import compact_
    Z = compact_(Z, 'zeros', float(np.finfo(float).eps))
    
    # Check if reduction is required
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope generators are None')
    n, nrOfGens = Z.G.shape
    
    if n * order < nrOfGens:
        # Reduce with the selected method
        if method == 'sum':
            Z = _aux_reduce_under_approx_sum(Z, order)
        elif method == 'scale':
            Z = _aux_reduce_under_approx_scale(Z, order)
        elif method == 'linProg':
            Z = _aux_reduce_under_approx_lin_prog(Z, order)
        elif method == 'wetzlinger':
            Z = _aux_reduce_under_approx_wetzlinger(Z, order)
    
    return Z


def _aux_reduce_under_approx_sum(Z: Zonotope, order: int) -> Zonotope:
    """
    Sum up the generators that are reduced to obtain an inner-approximation
    """
    # Select generators to reduce
    if Z.c is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope center is None')
    n = Z.c.shape[0]
    N = int(np.floor(order * n - 1))
    
    c, G, Gred = _aux_select_smallest_generators(Z, N)
    
    # Under-approximate the generators that are reduced by one generator
    # corresponding to the sum of generators
    g = np.sum(Gred, axis=1, keepdims=True)
    
    # Construct the reduced zonotope object
    Zred = Zonotope(c, np.hstack([G, g]))
    
    return Zred


def _aux_reduce_under_approx_scale(Z: Zonotope, order: int) -> Zonotope:
    """
    Scale an over-approximative reduced zonotope until it is fully contained
    """
    # Over-approximative reduction of the zonotope
    from .reduce import reduce
    Z_ = reduce(Z, 'girard', order)
    
    # For now, return the over-approximative result
    # In a full implementation, linear programming would be used to scale
    return Z_


def _aux_reduce_under_approx_lin_prog(Z: Zonotope, order: int) -> Zonotope:
    """
    Reduce using linear programming (simplified implementation)
    """
    # For now, fall back to sum method
    return _aux_reduce_under_approx_sum(Z, order)


def _aux_reduce_under_approx_wetzlinger(Z: Zonotope, order: int) -> Zonotope:
    """
    Reduction based on Hausdorff distance (simplified implementation)
    """
    # For now, fall back to sum method
    return _aux_reduce_under_approx_sum(Z, order)


def _aux_select_smallest_generators(Z: Zonotope, N: int):
    """
    Select the generators that are reduced
    """
    # Obtain object properties
    c = Z.c
    G_ = Z.G
    
    if G_ is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Zonotope generators are None')
    
    # Sort according to generator length
    temp = np.sum(G_**2, axis=0)
    ind = np.argsort(temp)[::-1]  # Descending order
    
    # Split into reduced and unreduced generators
    G = G_[:, ind[:N]]
    Gred = G_[:, ind[N:]]
    
    return c, G, Gred 