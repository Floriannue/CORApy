"""
calcSquaredE - computes the multiplicative E1' * E2

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Union
from .calcSquaredGInd import calcSquaredGInd


def calcSquaredE(E1: torch.Tensor, E2: torch.Tensor, isEqual: bool) -> torch.Tensor:
    """
    Compute the multiplicative E1' * E2.
    Internal to nn - only works with torch tensors.
    
    Args:
        E1: exponential matrix 1 (torch tensor)
        E2: exponential matrix 2 (torch tensor)
        isEqual: whether they are equal for optimizations
        
    Returns:
        E_quad: result of E1' * E2 as row vector (torch tensor)
        
    See also: polyZonotope/quadMap, nnHelper/calcSquared
    """
    # E1, E2 exponential matrices; calculate E1'*E2
    if E1.numel() > 0 and E2.numel() > 0:
        # Convert to numpy for calcSquaredGInd (it uses numpy indexing logic)
        E1_np = E1.cpu().numpy()
        E2_np = E2.cpu().numpy()
        G1_ind, G2_ind, G1_ind2, G2_ind2 = calcSquaredGInd(E1_np[0, :], E2_np[0, :], isEqual)
        # Use torch operations for the result
        E_quad = torch.cat([E1[:, G1_ind] + E2[:, G2_ind], E1[:, G1_ind2] + E2[:, G2_ind2]], dim=1)
    else:
        device = E1.device if isinstance(E1, torch.Tensor) else (E2.device if isinstance(E2, torch.Tensor) else torch.device('cpu'))
        dtype = E1.dtype if isinstance(E1, torch.Tensor) else (E2.dtype if isinstance(E2, torch.Tensor) else torch.long)
        E_quad = torch.empty((0, 0), dtype=dtype, device=device)
    
    return E_quad
