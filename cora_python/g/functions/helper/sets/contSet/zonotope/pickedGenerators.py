"""
pickedGenerators - Selects generators to be reduced and sorts the
    reduced generators

Syntax:
    [c, Gunred, Gred] = pickedGenerators(Z,order)

Inputs:
    Z - zonotope object
    order - desired order of the zonotope

Outputs:
    c - center of reduced zonotope
    Gunred - generators that are not reduced
    Gred - generators that are reduced
    indRed - indices that are reduced

Other m-files required: none
Subfunctions: none
MAT-files required: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       11-October-2017 (MATLAB)
Last update:   28-October-2017 (MATLAB)
               14-March-2019 (vector norm exchanged, remove sort, MATLAB)
               27-August-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonotope import Zonotope


def pickedGenerators(Z: 'Zonotope', order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects generators to be reduced and sorts the reduced generators
    
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
            # compute metric of generators
            # h = vecnorm(G,1,1) - vecnorm(G,Inf,1)
            h = np.linalg.norm(G, ord=1, axis=0) - np.linalg.norm(G, ord=np.inf, axis=0)
            
            # number of generators that are not reduced
            nUnreduced = int(np.floor(d * (order - 1)))
            # number of generators that are reduced
            nReduced = nrOfGens - nUnreduced
            
            # pick generators with smallest h values to be reduced
            # [~,indRed] = mink(h,nReduced);
            indRed = np.argsort(h)[:nReduced]
            Gred = G[:, indRed]
            
            # unreduced generators
            # indRemain = setdiff(1:nrOfGens, indRed);
            indRemain = np.setdiff1d(np.arange(nrOfGens), indRed)
            Gunred = G[:, indRemain]
        else:
            Gunred = G
    
    return c, Gunred, Gred, indRed 