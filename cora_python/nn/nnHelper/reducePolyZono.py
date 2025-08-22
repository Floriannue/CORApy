"""
reducePolyZono - reduce the number of generators of a polynomial zonotope, 
   where we exploit that an interval remainder is added when reducing 
   with Girards method

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple, Optional


def reducePolyZono(c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, 
                   id_: np.ndarray, nrGen: int, S: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce the number of generators of a polynomial zonotope.
    
    Args:
        c: center of polyZonotope
        G: dep. generator of polyZonotope
        GI: indep. generator of polyZonotope
        E: exponential matrix of polyZonotope
        id_: ids
        nrGen: number of generators
        S: sensitivity (optional, default: 1)
        
    Returns:
        c, G, GI, E, id_, d: reduced polynomial zonotope
        
    See also: -
    """
    if S is None:
        S = 1
    
    d = np.zeros_like(c)
    
    if nrGen < G.shape[1] + GI.shape[1]:
        # extract dimensions
        N = len(c)
        P = G.shape[1]
        Q = GI.shape[1]
        order = nrGen / N
        
        # number of generators that stay unreduced (N generators are added again
        # after reduction)
        K = max(0, int(np.floor(N * order - N)))
        
        # check if it is necessary to reduce the order
        if P + Q > N * order and K >= 0:
            
            # concatenate all generators, weighted by sensitivity
            SG = S * np.hstack([G, GI])
            
            # half the generator length for exponents that are all even
            ind_even = ~np.any(E % 2, axis=0)
            SG[:, ind_even] = 0.5 * SG[:, ind_even]
            
            # calculate the length of the generator vectors with a special metric
            len_vals = np.sum(SG**2, axis=0)
            
            # determine the smallest generators (= generators that are removed)
            ind_smallest = np.argsort(len_vals)[::-1]  # descending order
            ind_smallest = ind_smallest[K:]
            
            # split the indices into the ones for dependent and independent
            # generators
            ind_dep = ind_smallest[ind_smallest < P]
            ind_ind = ind_smallest[ind_smallest >= P]
            ind_ind = ind_ind - P
            
            # construct a zonotope from the generators that are removed
            G_rem = G[:, ind_dep]
            GI_rem = GI[:, ind_ind]
            c_red = np.zeros((N, 1))
            
            # half generators with all even exponents
            ind_even_rem = ind_even[ind_dep]
            G_rem[:, ind_even_rem] = 0.5 * G_rem[:, ind_even_rem]
            c_red = c_red + np.sum(0.5 * G_rem[:, ind_even_rem], axis=1, keepdims=True)
            
            # remove the generators that got reduced from the generator matrices
            G = np.delete(G, ind_dep, axis=1)
            E = np.delete(E, ind_dep, axis=1)
            GI = np.delete(GI, ind_ind, axis=1)
            
            # add shifted center
            c = c + c_red
            
            # box over-approximation as approx error
            d = np.sum(np.abs(np.hstack([G_rem, GI_rem])), axis=1, keepdims=True)
        
        # remove all exponent vector dimensions that have no entries
        ind = np.sum(E, axis=1) > 0
        E = E[ind, :]
        id_ = id_[ind]
    
    return c, G, GI, E, id_, d
