"""
pickedGeneratorsFast - Selects generators to be reduced without sorting

Syntax:
    [c, Gunred, Gred] = pickedGeneratorsFast(Z,order)

Inputs:
    Z - zonotope object
    order - desired order of the zonotope

Outputs:
    c - center of reduced zonotope
    Gunred - generators that are not reduced
    Gred - generators that are reduced
    indRed - indices that are reduced

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       11-October-2017 (MATLAB)
Last update:   28-October-2017 (MATLAB)
                14-March-2019 (vector norm exchanged, remove sort, MATLAB)
                27-August-2019 (MATLAB)
                20-July-2023 (TL, split mink/maxk, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonotope import Zonotope


def pickedGeneratorsFast(Z: 'Zonotope', order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects generators to be reduced without sorting (fast version)
    
    Args:
        Z: zonotope object
        order: desired order of the zonotope
        
    Returns:
        c: center of reduced zonotope
        Gunred: generators that are not reduced
        Gred: generators that are reduced
        indRed: indices that are reduced
    """
    from .nonzeroFilter import nonzeroFilter
    
    # center
    c = Z.c
    
    # extract generator matrix
    G = Z.G
    
    # default values
    Gunred = np.zeros((G.shape[0], 0))
    Gred = np.zeros((G.shape[0], 0))
    indRed = np.array([], dtype=int)
    
    if G.size > 0:
        # delete zero-length generators
        G = nonzeroFilter(G)
        
        # number of generators
        d, nrOfGens = G.shape
        
        # only reduce if zonotope order is greater than the desired order
        if nrOfGens > d * order:
            # number of generators that are not reduced
            nUnreduced = int(np.floor(d * (order - 1)))
            # number of generators that are reduced
            nReduced = nrOfGens - nUnreduced
            
            if nReduced == nrOfGens:
                # all generators are reduced
                Gred = G
                # indRed remains empty (not set in MATLAB either)
            else:
                # compute metric of generators
                # MATLAB: h = vecnorm(G,1,1) - vecnorm(G,Inf,1);
                h = np.linalg.norm(G, ord=1, axis=0) - np.linalg.norm(G, ord=np.inf, axis=0)
                
                if nReduced < nUnreduced:
                    # pick generators with smallest h values to be reduced
                    # MATLAB: [~,indRed] = mink(h,nReduced);
                    indRed = np.argpartition(h, nReduced - 1)[:nReduced]
                    indRed = indRed[np.argsort(h[indRed])]  # Sort by value
                    
                    # Create boolean index arrays
                    idxRed = np.zeros(nrOfGens, dtype=bool)
                    idxRed[indRed] = True
                    idxUnred = ~idxRed
                    
                else:
                    # pick generators with largest h values to be kept
                    # MATLAB: [~,indUnred] = maxk(fliplr(h),nUnreduced);
                    # fliplr reverses the array, then maxk gets largest, then we reverse indices
                    h_flipped = np.flip(h)
                    # Get indices of largest nUnreduced values in flipped array
                    indUnred_flipped = np.argpartition(h_flipped, len(h_flipped) - nUnreduced)[-nUnreduced:]
                    indUnred_flipped = indUnred_flipped[np.argsort(h_flipped[indUnred_flipped])]
                    # MATLAB: indUnred = nrOfGens - indUnred + 1; % maintain ordering
                    # MATLAB uses 1-based indexing: if flipped_idx is k (1-based), original is nrOfGens - k + 1
                    # Python uses 0-based: if flipped_idx is k (0-based), original is nrOfGens - 1 - k
                    indUnred = nrOfGens - 1 - indUnred_flipped  # Convert to original indices (0-based)
                    indUnred = np.sort(indUnred)  # Maintain ordering
                    
                    # Create boolean index arrays
                    idxUnred = np.zeros(nrOfGens, dtype=bool)
                    idxUnred[indUnred] = True
                    idxRed = ~idxUnred
                    
                    # Note: In MATLAB, indRed is NOT set in this branch (remains empty)
                    # We keep indRed empty to match MATLAB behavior
                
                # split G accordingly
                Gred = G[:, idxRed]
                Gunred = G[:, idxUnred]
        else:
            # no reduction
            Gunred = G
    
    return c, Gunred, Gred, indRed
