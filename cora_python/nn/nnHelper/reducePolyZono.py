"""
reducePolyZono - reduce the number of generators of a polynomial zonotope, 
   where we exploit that an interval remainder is added when reducing 
   with Girards method

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


def reducePolyZono(c: torch.Tensor, G: torch.Tensor, 
                   GI: torch.Tensor, E: torch.Tensor, 
                   id_: torch.Tensor, nrGen: int, 
                   S: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduce the number of generators of a polynomial zonotope.
    Internal to nn - only works with torch tensors.
    
    Args:
        c: center of polyZonotope (torch tensor)
        G: dep. generator of polyZonotope (torch tensor)
        GI: indep. generator of polyZonotope (torch tensor)
        E: exponential matrix of polyZonotope (torch tensor)
        id_: ids (torch tensor)
        nrGen: number of generators
        S: sensitivity (optional, default: 1) (torch tensor)
        
    Returns:
        c, G, GI, E, id_, d: reduced polynomial zonotope (torch tensors)
        
    See also: -
    """
    if S is None:
        S = torch.tensor(1.0, dtype=c.dtype, device=c.device)
    elif not isinstance(S, torch.Tensor):
        S = torch.tensor(S, dtype=c.dtype, device=c.device)
    
    device = c.device
    dtype = c.dtype
    d = torch.zeros_like(c)
    
    if nrGen < G.shape[1] + GI.shape[1]:
        # extract dimensions
        N = len(c)
        P = G.shape[1]
        Q = GI.shape[1]
        order = nrGen / N
        
        # number of generators that stay unreduced (N generators are added again
        # after reduction)
        K = max(0, int(torch.floor(torch.tensor(N * order - N, dtype=dtype, device=device)).item()))
        
        # check if it is necessary to reduce the order
        if P + Q > N * order and K >= 0:
            # Use torch operations only
            # concatenate all generators, weighted by sensitivity
            SG = S * torch.cat([G, GI], dim=1)
            
            # half the generator length for exponents that are all even
            ind_even = ~torch.any(E % 2, dim=0)
            SG[:, ind_even] = 0.5 * SG[:, ind_even]
            
            # calculate the length of the generator vectors with a special metric
            len_vals = torch.sum(SG**2, dim=0)
            
            # determine the smallest generators (= generators that are removed)
            ind_smallest = torch.argsort(len_vals, descending=True)  # descending order
            ind_smallest = ind_smallest[K:]
            
            # split the indices into the ones for dependent and independent
            # generators
            ind_dep = ind_smallest[ind_smallest < P]
            ind_ind = ind_smallest[ind_smallest >= P]
            ind_ind = ind_ind - P
            
            # construct a zonotope from the generators that are removed
            G_rem = G[:, ind_dep]
            GI_rem = GI[:, ind_ind]
            c_red = torch.zeros((N, 1), dtype=dtype, device=device)
            
            # half generators with all even exponents
            ind_even_rem = ind_even[ind_dep]
            G_rem[:, ind_even_rem] = 0.5 * G_rem[:, ind_even_rem]
            c_red = c_red + torch.sum(0.5 * G_rem[:, ind_even_rem], dim=1, keepdims=True)
            
            # remove the generators that got reduced from the generator matrices
            # Use torch indexing to remove columns
            keep_dep = torch.ones(P, dtype=torch.bool, device=device)
            keep_dep[ind_dep] = False
            G = G[:, keep_dep]
            E = E[:, keep_dep]
            
            keep_ind = torch.ones(Q, dtype=torch.bool, device=device)
            keep_ind[ind_ind] = False
            GI = GI[:, keep_ind]
            
            # add shifted center
            c = c + c_red
            
            # box over-approximation as approx error
            d = torch.sum(torch.abs(torch.cat([G_rem, GI_rem], dim=1)), dim=1, keepdims=True)
        
        # remove all exponent vector dimensions that have no entries
        ind = torch.sum(E, dim=1) > 0
        E = E[ind, :]
        id_ = id_[ind]
    
    return c, G, GI, E, id_, d
