"""
compact_ - removes redundancies in the representation of a polynomial
   zonotope, such as zero-length generators or unused dependent factors

Syntax:
   pZ = compact_(pZ)
   pZ = compact_(pZ,method)
   pZ = compact_(pZ,method,tol)

Inputs:
   pZ - polyZonotope object
   method - method for redundancy removal
            'states': remove redundancies in dependent generator matrix
            'exponentMatrix': remove redundancies in exponent matrix
            'all' (default): all of the above in succession
   tol - tolerance

Outputs:
   pZ - polyZonotope object without redundancies

Example: 
   pZ = polyZonotope([1;2],[1 3 1 -1 0;0 1 1 1 2], ...
                     [],[1 0 1 0 1;0 1 2 0 2])
   compact(pZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/compact, conPolyZono/compact_, zonotope/compact_

Authors:       Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       20-January-2020 (MATLAB)
Last update:   ---
Last revision: 30-July-2023 (MW, restructure, merge with deleteZeros) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


def compact_(pZ: 'PolyZonotope', method: str = 'all', tol: float = None) -> 'PolyZonotope':
    """
    Removes redundancies in the representation of a polynomial zonotope
    
    Args:
        pZ: polyZonotope object
        method: method for redundancy removal ('states', 'exponentMatrix', or 'all')
        tol: tolerance (default: eps)
        
    Returns:
        polyZonotope object without redundancies
    """
    if tol is None:
        tol = np.finfo(float).eps
    
    # Create a copy to avoid modifying the original
    # In MATLAB, this modifies the object in place, but we'll return a new one
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
    
    if method == 'states':
        # Remove empty generators (fast -> always)
        G, GI, E, id = _aux_removeRedundanciesGI(pZ.G, pZ.GI, pZ.E, pZ.id, tol)
        return PolyZonotope(pZ.c, G, GI, E, id)
    
    elif method == 'exponentMatrix':
        # Remove redundancies in the exponent matrix
        c, G, E, id = _aux_removeRedundanciesE(pZ.c, pZ.G, pZ.E, pZ.id)
        return PolyZonotope(c, G, pZ.GI, E, id)
    
    elif method == 'all':
        # Remove empty generators (fast -> always)
        G, GI, E, id = _aux_removeRedundanciesGI(pZ.G, pZ.GI, pZ.E, pZ.id, tol)
        # Remove redundancies in the exponent matrix
        c, G, E, id = _aux_removeRedundanciesE(pZ.c, G, E, id)
        return PolyZonotope(c, G, GI, E, id)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'states', 'exponentMatrix', or 'all'.")


def _aux_removeRedundanciesGI(G: np.ndarray, GI: np.ndarray, E: np.ndarray, id: np.ndarray, tol: float):
    """
    Remove redundancies in dependent and independent generator matrices
    
    Args:
        G: dependent generator matrix
        GI: independent generator matrix
        E: exponent matrix
        id: identifier vector
        tol: tolerance
        
    Returns:
        Tuple of (G, GI, E, id) with redundancies removed
    """
    # Indices with non-zero generators
    # MATLAB: idxD = any(G,1)
    idxD = np.any(G, axis=0) if G.size > 0 else np.array([], dtype=bool)
    # MATLAB: idxI = any(GI,1)
    idxI = np.any(GI, axis=0) if GI.size > 0 else np.array([], dtype=bool)
    
    # If all non-zero, skip
    if not (np.all(idxD) and np.all(idxI)):
        # Delete zero generators
        if G.size > 0:
            G = G[:, idxD]
            E = E[:, idxD] if E.size > 0 else E
        
        if GI.size > 0:
            GI = GI[:, idxI]
        
        # Delete zero exponents (rows of E that are all zero)
        # MATLAB: idxE = any(E,2)
        if E.size > 0:
            idxE = np.any(E, axis=1)
            if not np.all(idxE):
                E = E[idxE, :]
                if id.size > 0:
                    id = id[idxE]
    
    return G, GI, E, id


def _aux_removeRedundanciesE(c: np.ndarray, G: np.ndarray, E: np.ndarray, id: np.ndarray):
    """
    Remove redundancies in the exponent matrix
    
    Args:
        c: center vector
        G: dependent generator matrix
        E: exponent matrix
        id: identifier vector
        
    Returns:
        Tuple of (c, G, E, id) with redundancies removed
    """
    # Remove redundant exponent vectors
    if E.size > 0 and G.size > 0:
        from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents
        E, G = removeRedundantExponents(E, G)
    
    # Add all constant parts to the center
    # MATLAB: ind = find(sum(E,1) == 0)
    if E.size > 0:
        ind = np.where(np.sum(E, axis=0) == 0)[0]
        
        if len(ind) > 0:
            # MATLAB: c = c + sum(G(:,ind),2)
            c = c + np.sum(G[:, ind], axis=1, keepdims=True)
            # MATLAB: G(:,ind) = []; E(:,ind) = []
            G = np.delete(G, ind, axis=1)
            E = np.delete(E, ind, axis=1)
    
    # Remove empty rows from the exponent matrix
    # MATLAB: ind = find(sum(E,2) == 0)
    if E.size > 0:
        ind = np.where(np.sum(E, axis=1) == 0)[0]
        
        if len(ind) > 0:
            # MATLAB: E(ind,:) = []; id(ind) = []
            E = np.delete(E, ind, axis=0)
            if id.size > 0:
                id = np.delete(id, ind, axis=0)
    
    return c, G, E, id

