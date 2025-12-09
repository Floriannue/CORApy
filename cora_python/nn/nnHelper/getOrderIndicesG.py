"""
getOrderIndicesG - calculates the start and end indices for polynomial evaluation

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Tuple, Union


def getOrderIndicesG(G: Union[np.ndarray, torch.Tensor], order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the start and end indices for polynomial evaluation.
    Works with both numpy and torch (internal to nn, so torch is expected).
    
    Args:
        G: generator matrix (torch tensor or numpy array - torch expected internally)
        order: order of polynomial
        
    Returns:
        G_start: start indices of G^i, i \in 1:order (numpy array for compatibility)
        G_end: end indices of G^i, i \in 1:order (numpy array for compatibility)
        G_ext_start: start indices of extended G^i, i \in 1:order (numpy array for compatibility)
        G_ext_end: end indices of extended G^i, i \in 1:order (numpy array for compatibility)
            - where 'extended' refers to G5_ext = [G2_ext, G3_ext, G5]
            
    See also: nnActivationLayer/evaluatePolyZonotope
    """
    # init - return numpy arrays for compatibility (only used for indexing calculations)
    G_start = np.zeros(order, dtype=int)
    G_end = np.zeros(order, dtype=int)
    G_ext_start = np.zeros(order, dtype=int)
    G_ext_end = np.zeros(order, dtype=int)
    
    # init linear terms - works with both torch and numpy (only uses shape)
    n = G.shape[1]
    G_start[0] = 1
    G_end[0] = n
    G_ext_start[0] = 1
    G_ext_end[0] = n
    
    for o in range(2, order + 1):
        o1 = o // 2
        o2 = (o + 1) // 2  # equivalent to ceil(o/2)
        
        # init lengths
        o1_ext_len = G_ext_end[o1 - 1] - G_ext_start[o1 - 1] + 1
        o2_ext_len = G_ext_end[o2 - 1] - G_ext_start[o2 - 1] + 1
        
        if o1 == o2:
            # only requires computation of triangle
            n = o1_ext_len
            o_len = int(0.5 * n * (n + 1))
        else:
            # get full set of indices
            o_len = o1_ext_len * o2_ext_len
        
        # save results
        G_start[o - 1] = G_end[o - 2] + 1
        G_end[o - 1] = G_start[o - 1] + o_len - 1
        
        G_ext_start[o - 1] = G_ext_end[o - 2] + 1
        G_ext_end[o - 1] = G_ext_start[o - 1] + o1_ext_len + o2_ext_len + o_len - 1
    
    return G_start, G_end, G_ext_start, G_ext_end
