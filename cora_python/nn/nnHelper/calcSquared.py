"""
calcSquared - computes pZ_1 * pZ_2 with pZ_1, pZ_2 being multiplicatives 
   of a one-dimensional polyZonotope pZ:
   pZ_1 * pZ_2 =
        = (c1 + G1 + GI1)*(c2 + G2 + GI2)
        = c1*c2 + c1*G2 + c1*GI2
          + G1*c2 + G1*G2 + G2*GI2
          + GI1*c2 + GI1*G2 + GI1*GI2
        = (c1*c2 + sum(0.5[GI1*GI2](:, I)))
          % = c
          + (c1*G2 + G1*c2 + G1*G2)
          % = G
          + (c1*GI2 + GI2*c2 + 0.5[GI1*GI2](:, I) + [GI1*GI2](:, ~I) + G1*GI2 + GI2*G2)
          % = GI
          % with I ... indices of generators that need shifting

Note: (Half of) [GI1*GI2](:, I) appears in c & GI:
  Squaring independent Generators need shifting
  as only positive part is used afterwards.
  -> this will be corrected when adding the terms together

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Tuple, Union
from .calcSquaredG import calcSquaredG


def calcSquared(c1: Union[np.ndarray, torch.Tensor], G1: Union[np.ndarray, torch.Tensor], 
                GI1: Union[np.ndarray, torch.Tensor], E1: Union[np.ndarray, torch.Tensor],
                c2: Union[np.ndarray, torch.Tensor], G2: Union[np.ndarray, torch.Tensor], 
                GI2: Union[np.ndarray, torch.Tensor], E2: Union[np.ndarray, torch.Tensor],
                isEqual: bool) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Compute pZ_1 * pZ_2 with pZ_1, pZ_2 being multiplicatives of a one-dimensional polyZonotope pZ.
    Internal to nn - works with torch tensors.
    
    Args:
        c1: center of first polyZonotope (torch tensor expected internally)
        G1: dependent generators of first polyZonotope (torch tensor expected internally)
        GI1: independent generators of first polyZonotope (torch tensor expected internally)
        E1: exponential matrix of first polyZonotope (torch tensor expected internally)
        c2: center of second polyZonotope (torch tensor expected internally)
        G2: dependent generators of second polyZonotope (torch tensor expected internally)
        GI2: independent generators of second polyZonotope (torch tensor expected internally)
        E2: exponential matrix of second polyZonotope (torch tensor expected internally)
        isEqual: whether the two polyZonotopes are equal
        
    Returns:
        c: center of pZ^(i1+i2) (torch tensor)
        G: dependent generators of pZ^(i1+i2) (torch tensor)
        GI: independent generators of pZ^(i1+i2) (torch tensor)
        
    See also: polyZonotope/quadMap, nnHelper/calcSquaredG, nnHelper/calcSquaredE
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(c1, np.ndarray):
        c1 = torch.tensor(c1, dtype=torch.float32)
    if isinstance(G1, np.ndarray):
        G1 = torch.tensor(G1, dtype=torch.float32)
    if isinstance(GI1, np.ndarray):
        GI1 = torch.tensor(GI1, dtype=torch.float32)
    if isinstance(E1, np.ndarray):
        E1 = torch.tensor(E1, dtype=torch.int64)
    if isinstance(c2, np.ndarray):
        c2 = torch.tensor(c2, dtype=torch.float32)
    if isinstance(G2, np.ndarray):
        G2 = torch.tensor(G2, dtype=torch.float32)
    if isinstance(GI2, np.ndarray):
        GI2 = torch.tensor(GI2, dtype=torch.float32)
    if isinstance(E2, np.ndarray):
        E2 = torch.tensor(E2, dtype=torch.int64)
    
    device = c1.device
    dtype = c1.dtype
    
    G_quad = calcSquaredG(G1, G2, isEqual)
    GI_quad = calcSquaredG(GI1, GI2, isEqual)
    
    # construct squared parameters
    c = c1 * c2
    
    # See Note
    if isEqual:
        r = GI1.shape[1]
        GI_quad[:, :r] = 0.5 * GI_quad[:, :r]
        c = c + torch.sum(GI_quad[:, :r], dim=1, keepdim=True)
    
    # the same principle applies to G1, G2:
    # if all exponents are even and they become independent generators
    # after multiplying them with GI2, GI1, respectively.
    # except center does not need shifting as
    # G1, G2 only scale the independent generators of GI2, GI1.
    
    # G1 * GI2
    even_indices = torch.all(E1 % 2 == 0, dim=0)
    G1_ind = G1.clone()  # copy by value
    G1_ind[:, even_indices] = 0.5 * G1_ind[:, even_indices]
    G1GI2 = calcSquaredG(G1_ind, GI2)
    
    # GI1 * G2
    even_indices = torch.all(E2 % 2 == 0, dim=0)
    G2_ind = G2.clone()  # copy by value
    G2_ind[:, even_indices] = 0.5 * G2_ind[:, even_indices]
    GI1G2 = calcSquaredG(GI1, G2_ind)
    
    G = torch.cat([G1 * c2, c1 * G2, G_quad], dim=1)
    
    if isEqual:
        GI = torch.cat([GI1 * c2, c1 * GI1, GI_quad, 2 * GI1G2], dim=1)
    else:
        GI = torch.cat([GI1 * c2, c1 * GI2, GI_quad, G1GI2, GI1G2], dim=1)
    
    return c, G, GI
