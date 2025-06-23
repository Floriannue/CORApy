"""
compact_ - returns equal zonotope in minimal representation

Syntax:
   Z = compact_(Z, method, tol)

Inputs:
   Z - zonotope object
   method - method for redundancy removal
            'zeros': removes all generators from a zonotope
                 with zeros in all dimensions, so that every generator
                 the resulting zonotope has at least one non-zero entry
            'all': combines aligned generators to a single generator;
                 a tolerance is used to determine alignment, so this
                 function does not necessarily return an
                 over-approximation of the original zonotope---for this,
                 use zonotope/reduce instead
   tol - tolerance

Outputs:
   Z - zonotope object

Example:
   Z1 = zonotope([0;0],[1 0 -2 0 3 4; 0 0 1 0 -2 1]);
   Z2 = compact(Z1);
   
   plot(Z1); hold on;
   plot(Z2,[1,2],'r');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/compact, zonotope/reduce

Authors: Mark Wetzlinger, Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 15-January-2009 (MATLAB)
Last update: 27-August-2019
             05-October-2024 (MW, remove superfluous 'aligned', rewrite deleteAligned)
Last revision: 29-July-2023 (MW, merged from deleteZeros/deleteAligned)
Python translation: 2025
"""

import numpy as np
from typing import Union
from .zonotope import Zonotope

def compact_(Z, method: str = 'zeros', tol: float = None) -> Zonotope:
    """
    Returns equal zonotope in minimal representation
    
    Args:
        Z: zonotope object
        method: method for redundancy removal ('zeros' or 'all')
        tol: tolerance (default: eps for 'zeros', 1e-3 for 'all')
        
    Returns:
        zonotope object in minimal representation
    """
    if tol is None:
        if method == 'zeros':
            tol = np.finfo(float).eps
        else:  # method == 'all'
            tol = 1e-3
    
    if method == 'zeros':
        return _aux_deleteZeros(Z, tol)
    elif method == 'all':
        return _aux_deleteAligned(Z, tol)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zeros' or 'all'.")


def _aux_deleteZeros(Z, tol: float) -> Zonotope:
    """
    Filter zero generators
    
    Args:
        Z: zonotope object
        tol: tolerance
        
    Returns:
        zonotope object with zero generators removed
    """
    # Filter zero generators using nonzeroFilter
    G_filtered = _nonzeroFilter(Z.G, tol)
    
    # Create new zonotope with filtered generators
    return Zonotope(Z.c, G_filtered)


def _aux_deleteAligned(Z, tol: float) -> Zonotope:
    """
    Delete aligned generators by combining them
    
    Args:
        Z: zonotope object
        tol: tolerance
        
    Returns:
        zonotope object with aligned generators combined
    """
    # Delete zero-generators first
    G = _nonzeroFilter(Z.G, tol)
    
    # Quick exit for 1D case
    if Z.dim() == 1:
        G_new = np.array([[np.sum(np.abs(G))]])
        return Zonotope(Z.c, G_new)
    
    # Check if no generators left
    if G.shape[1] == 0:
        return Zonotope(Z.c, G)
    
    # Normalize generators
    G_norms = np.linalg.norm(G, axis=0)
    # Avoid division by zero (should not happen after nonzeroFilter, but be safe)
    valid_gens = G_norms > tol
    if not np.any(valid_gens):
        return Zonotope(Z.c, np.empty((Z.dim(), 0)))
    
    G = G[:, valid_gens]
    G_norms = G_norms[valid_gens]
    G_norm = G / G_norms[np.newaxis, :]
    
    nrGen = G.shape[1]
    
    # Find aligned generators: since the generators are normalized, aligned
    # generators must have a dot product of 1 (parallel) or -1
    # (anti-parallel) with the generator in question; generators are then
    # unified by addition (taking 1/-1 into account)
    idxKeep = np.ones(nrGen, dtype=bool)
    
    for i in range(nrGen):
        # Only check those that have not yet been removed in a previous iteration
        if idxKeep[i]:
            # Compute dot product with all generators
            dotprod = G_norm[:, i].T @ G_norm
            
            # Check for which pairs the value is 1 or -1
            idxParallel = np.abs(dotprod - 1) <= tol
            idxAntiParallel = np.abs(dotprod + 1) <= tol
            
            # Add all generators with dotprod = 1 to ith generator,
            # subtract all generators with dotprod = -1 from ith generator
            # (note: dotprod with itself is 1, therefore no extra "+G(:,i)")
            if np.sum(idxParallel) > 1 or np.any(idxAntiParallel):
                G[:, i] = np.sum(G[:, idxParallel], axis=1) - np.sum(G[:, idxAntiParallel], axis=1)
                
                # ith generator and all (anti-)parallel generators have been
                # subsumed -> remove them from further checks; keep ith
                # generator for final removal after loop
                idxParallel[i] = False
                idxKeep = idxKeep & ~idxParallel & ~idxAntiParallel
    
    # Lengthening has already taken place in the loop above, so we only
    # need to remove all subsumed generators
    G_final = G[:, idxKeep]
    
    return Zonotope(Z.c, G_final)


def _nonzeroFilter(G: np.ndarray, tol: float = None) -> np.ndarray:
    """
    Filters out generators of length 0
    
    Args:
        G: matrix of generators
        tol: tolerance (optional)
        
    Returns:
        reduced matrix of generators
    """
    # Delete zero-generators (any non-zero entry in a column)
    G_filtered = G[:, np.any(G != 0, axis=0)]
    
    if tol is not None:
        # Also remove generators with norm below tolerance
        G_norms = np.linalg.norm(G_filtered, axis=0)
        G_filtered = G_filtered[:, G_norms > tol]
    
    return G_filtered 