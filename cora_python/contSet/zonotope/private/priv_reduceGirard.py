"""
priv_reduceGirard - Girard's method for zonotope order reduction (Sec. 4 in [2])

This is the most commonly used reduction method.

Authors:       Matthias Althoff
Written:       24-January-2007 
Last update:   15-September-2007
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

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
    if G.shape[1] <= order * n or G.shape[1] == 0:
        return Z.copy()
    
    # Remove zero-length generators (nonzeroFilter equivalent)
    G = G[:, np.any(G != 0, axis=0)]
    
    # Number of generators
    nrOfGens = G.shape[1]
    
    # Only reduce if zonotope order is greater than the desired order
    if nrOfGens > n * order:
        # Number of generators that are not reduced
        nUnreduced = int(np.floor(n * (order - 1)))
        # Number of generators that are reduced
        nReduced = nrOfGens - nUnreduced
        
        if nReduced == nrOfGens:
            # All generators are reduced
            Gred = G
            Gunred = np.zeros((n, 0))
        else:
            # Compute metric of generators: h = vecnorm(G,1,1) - vecnorm(G,Inf,1)
            h = np.linalg.norm(G, ord=1, axis=0) - np.linalg.norm(G, ord=np.inf, axis=0)
            
            if nReduced < nUnreduced:
                # Pick generators with smallest h values to be reduced
                indRed = np.argsort(h)[:nReduced]
                idxRed = np.zeros(nrOfGens, dtype=bool)
                idxRed[indRed] = True
                idxUnred = ~idxRed
            else:
                # Pick generators with largest h values to be kept
                indUnred = np.argsort(h)[-(nUnreduced):]
                idxUnred = np.zeros(nrOfGens, dtype=bool)
                idxUnred[indUnred] = True
                idxRed = ~idxUnred
            
            # Split G accordingly
            Gred = G[:, idxRed]
            Gunred = G[:, idxUnred]
    else:
        # No reduction
        Gunred = G
        Gred = np.zeros((n, 0))
    
    # Box remaining generators: sum of absolute values
    if Gred.size > 0:
        d = np.sum(np.abs(Gred), axis=1, keepdims=True)
        Gbox = np.diag(d.flatten())
    else:
        Gbox = np.zeros((n, 0))
    
    # Build reduced zonotope
    if Gunred.size > 0 and Gbox.size > 0:
        G_new = np.hstack([Gunred, Gbox])
    elif Gunred.size > 0:
        G_new = Gunred
    elif Gbox.size > 0:
        G_new = Gbox
    else:
        G_new = np.zeros((n, 0))
    
    return Zonotope(Z.c, G_new) 