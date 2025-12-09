"""
calcSquaredG - computes the multiplicative G1' * G2

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Optional, Union


def calcSquaredG(G1: Union[np.ndarray, torch.Tensor], G2: Union[np.ndarray, torch.Tensor], 
                 isEqual: Optional[bool] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the multiplicative G1' * G2.
    Internal to nn - works with torch tensors.
    
    Args:
        G1: row vector of generator matrix 1 (torch tensor expected internally)
        G2: row vector of generator matrix 2 (torch tensor expected internally)
        isEqual: whether they are equal for optimizations (default: False)
        
    Returns:
        G_quad: result of G1' * G2 as row vector (torch tensor)
        
    See also: polyZonotope/quadMap, nnHelper/calcSquared
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(G1, np.ndarray):
        G1 = torch.tensor(G1, dtype=torch.float32)
    if isinstance(G2, np.ndarray):
        G2 = torch.tensor(G2, dtype=torch.float32)
    
    if isEqual is None:
        isEqual = False
    
    device = G1.device
    dtype = G1.dtype
    
    if G1.numel() > 0 and G2.numel() > 0:
        if isEqual:
            temp = G1.T @ G2
            
            # we can ignore the left lower triangle in this case
            # as it's the same as the right upper triangle
            # -> double right upper triangle
            n = G1.shape[1]
            G_quad = torch.zeros((1, int(0.5 * n * (n + 1))), dtype=dtype, device=device)
            cnt = n
            
            for i in range(n - 1):
                G_quad[0, i] = temp[i, i]
                G_quad[0, cnt:cnt + n - i - 1] = 2 * temp[i, i + 1:n]
                cnt = cnt + n - i - 1
            G_quad[0, n - 1] = temp[-1, -1]
        else:
            # calculate all values
            G_quad = G1.T @ G2
            G_quad = G_quad.reshape(1, -1)  # row vector
    else:
        G_quad = torch.empty((1, 0), dtype=dtype, device=device)
    
    return G_quad
