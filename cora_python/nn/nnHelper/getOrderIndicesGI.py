"""
getOrderIndicesGI - calculates the start and end indices for polynomial evaluation

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Tuple, Union
from .getOrderIndicesG import getOrderIndicesG


def getOrderIndicesGI(GI: Union[np.ndarray, torch.Tensor], G: Union[np.ndarray, torch.Tensor], order: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the start and end indices for polynomial evaluation.
    Works with both numpy and torch (internal to nn, so torch is expected).
    
    Args:
        GI: ind. generator matrix (torch tensor or numpy array - torch expected internally)
        G: generator matrix (torch tensor or numpy array - torch expected internally)
        order: order of polynomial
        
    Returns:
        GI_start: start indices of GI^i, i in 1:order (torch tensor)
        GI_end: end indices of GI^i, i in 1:order (torch tensor)
        GI_ext_start: start indices of extended GI^i, i in 1:order (torch tensor)
        GI_ext_end: end indices of extended GI^i, i in 1:order (torch tensor)
            - where 'extended' refers to GI5_ext = [GI2_ext, GI3_ext, GI5]
            
    See also: nnActivationLayer/evaluatePolyZonotope
    """
    # init - use torch internally (internal to nn)
    device = GI.device if isinstance(GI, torch.Tensor) else (G.device if isinstance(G, torch.Tensor) else torch.device('cpu'))
    GI_start = torch.zeros(order, dtype=torch.long, device=device)
    GI_end = torch.zeros(order, dtype=torch.long, device=device)
    GI_ext_start = torch.zeros(order, dtype=torch.long, device=device)
    GI_ext_end = torch.zeros(order, dtype=torch.long, device=device)
    
    # init linear terms - works with both torch and numpy (only uses shape)
    n = GI.shape[1]
    GI_start[0] = 1
    GI_end[0] = n
    GI_ext_start[0] = 1
    GI_ext_end[0] = n
    
    # get lengths of G
    _, _, G_ext_start, G_ext_end = getOrderIndicesG(G, order)
    
    for o in range(2, order + 1):
        o1 = o // 2
        o2 = (o + 1) // 2  # equivalent to ceil(o/2)
        
        # init lengths - use torch operations
        o1_ext_len = (GI_ext_end[o1 - 1] - GI_ext_start[o1 - 1] + 1).item() if isinstance(GI_ext_end[o1 - 1], torch.Tensor) else int(GI_ext_end[o1 - 1] - GI_ext_start[o1 - 1] + 1)
        o2_ext_len = (GI_ext_end[o2 - 1] - GI_ext_start[o2 - 1] + 1).item() if isinstance(GI_ext_end[o2 - 1], torch.Tensor) else int(GI_ext_end[o2 - 1] - GI_ext_start[o2 - 1] + 1)
        
        G_o1_ext_len = (G_ext_end[o1 - 1] - G_ext_start[o1 - 1] + 1).item() if isinstance(G_ext_end[o1 - 1], torch.Tensor) else int(G_ext_end[o1 - 1] - G_ext_start[o1 - 1] + 1)
        G_o2_ext_len = (G_ext_end[o2 - 1] - G_ext_start[o2 - 1] + 1).item() if isinstance(G_ext_end[o2 - 1], torch.Tensor) else int(G_ext_end[o2 - 1] - G_ext_start[o2 - 1] + 1)
        
        if o1 == o2:
            # only requires computation of triangle
            n_val = o1_ext_len
            o_len = int(0.5 * n_val * (n_val + 1) + G_o1_ext_len * o2_ext_len)
        else:
            # get full set of indices
            o_len = o1_ext_len * o2_ext_len + G_o1_ext_len * o2_ext_len + G_o2_ext_len * o1_ext_len
        
        # save results - use torch operations
        GI_start[o - 1] = GI_end[o - 2] + 1
        GI_end[o - 1] = GI_start[o - 1] + o_len
        GI_ext_start[o - 1] = GI_ext_end[o - 2] + 1
        GI_ext_end[o - 1] = GI_ext_start[o - 1] + o1_ext_len + o2_ext_len + o_len - 1
    
    return GI_start, GI_end, GI_ext_start, GI_ext_end
