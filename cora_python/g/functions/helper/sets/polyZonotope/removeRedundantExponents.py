"""
removeRedundantExponents - add up all generators that belong to terms
   with identical exponents

Syntax:
   [Enew, Gnew] = removeRedundantExponents(E,G)

Inputs:
   E - matrix containing the exponent vectors
   G - generator matrix

Outputs:
   Enew - modified exponent matrix
   Gnew - modified generator matrix

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Niklas Kochdumper, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       25-June-2018 (MATLAB)
Last update:   21-April-2020 (remove zero-length generators, MATLAB)
               23-June-2022 (performance optimizations, MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Tuple


def removeRedundantExponents(E: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add up all generators that belong to terms with identical exponents
    
    Args:
        E: matrix containing the exponent vectors
        G: generator matrix
        
    Returns:
        Tuple of (Enew, Gnew) - modified exponent and generator matrices
    """
    
    # return directly if G is empty
    if G.size == 0:
        return E, G
    
    # remove zero-length generators
    idxD = np.any(G, axis=0)
    
    # skip if all non-zero
    if not np.all(idxD):
        if not np.any(idxD):
            # All generators are zero
            Enew = np.zeros((E.shape[0], 1))
            Gnew = np.zeros((G.shape[0], 1))
            return Enew, Gnew
        else:
            G = G[:, idxD]
            E = E[:, idxD]
    
    # create a deterministic random hash vector
    np.random.seed(0)  # to not interfere with the outside
    hashVec = np.random.rand(1, E.shape[0])
    hashMat = (hashVec @ E).T
    
    # sort the exponent vectors according to the hash value
    ind = np.argsort(hashMat.flatten())
    hashes = hashMat.flatten()[ind]
    
    # test if all (sorted) hashes are different
    if len(hashes) == 1:
        uniqueHashIdx = np.array([True])
    else:
        uniqueHashIdx = np.concatenate([
            hashes[:-1] != hashes[1:],  # successive different
            [True]  # last element is always unique in this context
        ]) & np.concatenate([
            [True],  # first element is always unique in this context
            hashes[:-1] != hashes[1:]  # predecessor different
        ])
    
    # if so, return directly
    numUnique = np.sum(uniqueHashIdx)
    if numUnique == E.shape[1]:
        return E, G
    
    # initialize new matrices
    Enew = np.zeros_like(E)
    Gnew = np.zeros_like(G)
    
    # copy unique hashes
    uniqueColumns = ind[uniqueHashIdx]
    Enew[:, :numUnique] = E[:, uniqueColumns]
    Gnew[:, :numUnique] = G[:, uniqueColumns]
    current = numUnique
    
    # continue with potential redundancies
    redundant_ind = ind[~uniqueHashIdx]
    
    if len(redundant_ind) == 0:
        # No redundancies found, truncate and return
        Enew = Enew[:, :current]
        Gnew = Gnew[:, :current]
        return Enew, Gnew
    
    E_red = E[:, redundant_ind]
    G_red = G[:, redundant_ind]
    hashMat_red = hashMat.flatten()[redundant_ind]
    
    # first entry
    exp_c = E_red[:, 0]
    hash_c = hashMat_red[0]
    Enew[:, current] = exp_c
    Gnew[:, current] = G_red[:, 0]
    
    # loop over all exponent vectors
    for i in range(1, len(redundant_ind)):
        hash_i = hashMat_red[i]
        exp_i = E_red[:, i]
        
        if hash_c == hash_i and np.all(exp_c == exp_i):
            Gnew[:, current] = Gnew[:, current] + G_red[:, i]
        else:
            current = current + 1
            exp_c = exp_i
            hash_c = hash_i
            Enew[:, current] = exp_c
            Gnew[:, current] = G_red[:, i]
    
    # truncate exponent and generator matrix
    Enew = Enew[:, :current + 1]
    Gnew = Gnew[:, :current + 1]
    
    return Enew, Gnew 