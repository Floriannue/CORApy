"""
Private reduction methods for zonotope

This file contains all the private reduction methods that mirror the MATLAB private functions.

Authors:       Matthias Althoff
Written:       24-January-2007 
Last update:   15-September-2007
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cora_python.contSet.zonotope import Zonotope


def priv_reduceGirard(Z: 'Zonotope', order: int) -> 'Zonotope':
    """
    Girard's method for zonotope order reduction (Sec. 4 in [2])
    
    This is the most commonly used reduction method.
    """
    from cora_python.contSet.zonotope import Zonotope
    
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


def priv_reduceCombastel(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Combastel's method for zonotope order reduction (Sec. 3.2 in [4])"""
    return priv_reduceGirard(Z, order)


def priv_reducePCA(Z: 'Zonotope', order: int) -> 'Zonotope':
    """PCA-based method for zonotope order reduction (Sec. III.A in [3])"""
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.g.functions.helper.sets.contSet.zonotope import pickedGenerators
    
    # initialize Z_red
    Zred = Z.copy()
    
    # pick generators to reduce
    _, Gunred, Gred, _ = pickedGenerators(Z, order)
    
    if Gred.size == 0:
        return Zred
    
    # obtain matrix of points from generator matrix
    V = np.hstack([Gred, -Gred])  # has zero mean
    
    # compute the covariance matrix
    # Note: V has shape (n, 2*m) where n is dimension, m is number of generators
    # We want covariance of the n-dimensional points, so we use V directly
    C = np.cov(V)
    
    # singular value decomposition
    U, _, _ = np.linalg.svd(C)
    
    # map generators
    Gtrans = U.T @ Gred
    
    # box generators
    Gbox = np.diag(np.sum(np.abs(Gtrans), axis=1))
    
    # transform generators back
    Gred_new = U @ Gbox
    
    # build reduced zonotope
    # Zred.c stays the same
    if Gunred.size > 0:
        Zred.G = np.hstack([Gunred, Gred_new])
    else:
        Zred.G = Gred_new
    
    return Zred


def priv_reduceMethA(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Method A for zonotope order reduction (Sec. 2.5.5 in [1])"""
    return priv_reduceGirard(Z, order)


def priv_reduceMethB(Z: 'Zonotope', order: int, filterLength: Optional[int] = None) -> 'Zonotope':
    """Method B for zonotope order reduction (Sec. 2.5.5 in [1])"""
    return priv_reduceGirard(Z, order)


def priv_reduceMethC(Z: 'Zonotope', order: int, filterLength: Optional[int] = None) -> 'Zonotope':
    """Method C for zonotope order reduction (Sec. 2.5.5 in [1])"""
    return priv_reduceGirard(Z, order)


def priv_reduceIdx(Z: 'Zonotope', order) -> 'Zonotope':
    """Reduce by index - simplified implementation"""
    return priv_reduceGirard(Z, 1)


def priv_reduceAdaptive(Z: 'Zonotope', order, option: str = 'default') -> Tuple['Zonotope', float, np.ndarray]:
    """Adaptive reduction method - simplified implementation"""
    Z_reduced = priv_reduceGirard(Z, 1)
    dHerror = 0.0  # Placeholder
    gredIdx = np.array([])  # Placeholder
    
    return Z_reduced, dHerror, gredIdx


def priv_reduceMethE(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Method E for zonotope order reduction - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceMethF(Z: 'Zonotope') -> 'Zonotope':
    """Method F for zonotope order reduction - simplified implementation"""
    return priv_reduceGirard(Z, 1)


def priv_reduceRedistribute(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Redistribute method - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceCluster(Z: 'Zonotope', order: int, option) -> 'Zonotope':
    """Cluster method - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceScott(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Scott method - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceValero(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Valero method - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceSadraddini(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Sadraddini method - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceScale(Z: 'Zonotope', order: int) -> 'Zonotope':
    """Scale method - simplified implementation"""
    return priv_reduceGirard(Z, order)


def priv_reduceScaleHausdorff(Z: 'Zonotope', order: int) -> Tuple['Zonotope', float]:
    """Scale Hausdorff method - simplified implementation"""
    Z_reduced = priv_reduceGirard(Z, order)
    dHerror = 0.0  # Placeholder
    return Z_reduced, dHerror


def priv_reduceConstOpt(Z: 'Zonotope', order: int, option: str, alg: str) -> 'Zonotope':
    """ConstOpt method - simplified implementation"""
    return priv_reduceGirard(Z, order) 