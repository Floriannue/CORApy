"""
compBoundsPolyZono - compute the lower and upper bound of a polynomial
   zonotope

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import torch
from typing import Tuple, Union


def compBoundsPolyZono(c: Union[np.ndarray, torch.Tensor], G: Union[np.ndarray, torch.Tensor], 
                       GI: Union[np.ndarray, torch.Tensor], E: Union[np.ndarray, torch.Tensor], 
                       ind: Union[np.ndarray, torch.Tensor], ind_: Union[np.ndarray, torch.Tensor], 
                       approx: bool) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Compute the lower and upper bound of a polynomial zonotope.
    Internal to nn - works with torch tensors.
    
    Args:
        c: center of polyZonotope in a dimension (torch tensor expected internally)
        G: corresponding dep. generator of polyZonotope as row vector (torch tensor expected internally)
        GI: corresponding indep. generator of polyZonotope as row vector (torch tensor expected internally)
        E: exponential matrix of polyZonotope (torch tensor expected internally)
        ind: all even indices (torch tensor expected internally)
        ind_: all odd indices (torch tensor expected internally)
        approx: whether to use approximation
        
    Returns:
        l, u: lower and upper bound (torch tensors)
        
    See also: -
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(c, np.ndarray):
        c = torch.tensor(c, dtype=torch.float32)
    if isinstance(G, np.ndarray):
        G = torch.tensor(G, dtype=torch.float32)
    if isinstance(GI, np.ndarray):
        GI = torch.tensor(GI, dtype=torch.float32)
    if isinstance(E, np.ndarray):
        E = torch.tensor(E, dtype=torch.int64)
    if isinstance(ind, np.ndarray):
        ind = torch.tensor(ind, dtype=torch.long)
    if isinstance(ind_, np.ndarray):
        ind_ = torch.tensor(ind_, dtype=torch.long)
    
    if approx:
        # using zonotope over-approximation - use torch
        # MATLAB: c_ = c + 0.5 * sum(G(:, ind), 2);
        c_ = c + 0.5 * torch.sum(G[:, ind], dim=1, keepdim=True)
        
        # MATLAB: l = c_ - sum(abs(0.5*G(:, ind)), 2) - sum(abs(G(:, ind_)), 2) - sum(abs(GI), 2);
        l = c_ - torch.sum(torch.abs(0.5 * G[:, ind]), dim=1, keepdim=True) - \
              torch.sum(torch.abs(G[:, ind_]), dim=1, keepdim=True) - \
              torch.sum(torch.abs(GI), dim=1, keepdim=True)
        u = c_ + torch.sum(torch.abs(0.5 * G[:, ind]), dim=1, keepdim=True) + \
              torch.sum(torch.abs(G[:, ind_]), dim=1, keepdim=True) + \
              torch.sum(torch.abs(GI), dim=1, keepdim=True)
    else:
        # tighter bounds using splitting - convert to numpy for PolyZonotope (external interface)
        from cora_python.contSet.polyZonotope import PolyZonotope
        
        # Convert to numpy for PolyZonotope constructor
        c_np = c.cpu().numpy() if isinstance(c, torch.Tensor) else c
        G_np = G.cpu().numpy() if isinstance(G, torch.Tensor) else G
        GI_np = GI.cpu().numpy() if isinstance(GI, torch.Tensor) else GI
        E_np = E.cpu().numpy() if isinstance(E, torch.Tensor) else E
        
        # Create polyZonotope object
        pZ = PolyZonotope(c_np, G_np, GI_np, E_np)
        
        # Compute interval using split method
        int_result = pZ.interval('split')
        
        # Extract bounds and convert back to torch
        device = c.device if isinstance(c, torch.Tensor) else torch.device('cpu')
        dtype = c.dtype if isinstance(c, torch.Tensor) else torch.float32
        l = torch.tensor(int_result.inf.reshape(-1, 1), dtype=dtype, device=device)
        u = torch.tensor(int_result.sup.reshape(-1, 1), dtype=dtype, device=device)
    
    return l, u
