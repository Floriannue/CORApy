"""
minMaxDiffOrder - compute the maximum and the minimum difference between the activation
function and a polynomial fit

Syntax:
    [diffl,diffu] = nnHelper.minMaxDiffOrder(coeffs, l, u, f, der1)

Inputs:
    coeffs - coefficients of polynomial
    l - lower bound of input domain
    u - upper bound of input domain
    f - function handle of activation function
    der1 - bounds for derivative of activation functions

Outputs:
    [diffl,diffu] - interval bounding the lower and upper error

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   31-August-2022 (adjust tol)
               30-May-2023 (output bounds)
               02-May-2025 (added maxPoints)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Union, Callable
from .getDerInterval import getDerInterval

def minMaxDiffOrder(coeffs: Union[np.ndarray, torch.Tensor], l: Union[float, np.ndarray, torch.Tensor], 
                   u: Union[float, np.ndarray, torch.Tensor], 
                   f: Callable, der1l: Union[float, np.ndarray, torch.Tensor], 
                   der1u: Union[float, np.ndarray, torch.Tensor]) -> tuple:
    """
    Compute the maximum and the minimum difference between the activation
    function and a polynomial fit.
    Internal to nn - works with torch tensors.
    
    Args:
        coeffs: coefficients of polynomial (torch tensor expected internally)
        l: lower bound of input domain (torch tensor expected internally)
        u: upper bound of input domain (torch tensor expected internally)
        f: function handle of activation function (works with torch)
        der1l, der1u: bounds for derivative of activation functions (torch tensor expected internally)
        
    Returns:
        Tuple of (diffl, diffu) interval bounding the lower and upper error (torch tensors)
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(coeffs, np.ndarray):
        coeffs = torch.tensor(coeffs, dtype=torch.float32)
    if isinstance(l, np.ndarray):
        l = torch.tensor(l, dtype=torch.float32)
    if isinstance(u, np.ndarray):
        u = torch.tensor(u, dtype=torch.float32)
    if isinstance(der1l, np.ndarray):
        der1l = torch.tensor(der1l, dtype=torch.float32)
    if isinstance(der1u, np.ndarray):
        der1u = torch.tensor(der1u, dtype=torch.float32)
    
    if not isinstance(coeffs, torch.Tensor):
        coeffs = torch.tensor(coeffs, dtype=torch.float32)
    if not isinstance(l, torch.Tensor):
        l = torch.tensor(l, dtype=torch.float32)
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(der1l, torch.Tensor):
        der1l = torch.tensor(der1l, dtype=torch.float32)
    if not isinstance(der1u, torch.Tensor):
        der1u = torch.tensor(der1u, dtype=torch.float32)
    
    device = coeffs.device
    dtype = coeffs.dtype
    
    tol = 1e-4
    minPoints = 1e4
    maxPoints = 5e9  # requires 40GB
    
    # Handle scalar vs tensor comparison
    l_val = l.item() if isinstance(l, torch.Tensor) and l.numel() == 1 else l
    u_val = u.item() if isinstance(u, torch.Tensor) and u.numel() == 1 else u
    
    if l_val == u_val:
        # compute exact result directly
        diff = f(l)  # f works with torch
        # Use torch.polyval equivalent (manual implementation)
        yp = torch.polyval(coeffs, l) if hasattr(torch, 'polyval') else _torch_polyval(coeffs, l)
        diff = diff - yp
        diffl = diff
        diffu = diff
        return diffl, diffu
    
    # calculate bounds for derivative of polynomial
    # getDerInterval expects numpy, so convert at boundary
    coeffs_np = coeffs.cpu().numpy()
    l_np = l_val if isinstance(l_val, (int, float)) else l.cpu().numpy()
    u_np = u_val if isinstance(u_val, (int, float)) else u.cpu().numpy()
    der2l, der2u = getDerInterval(coeffs_np, l_np, u_np)
    der2l = torch.tensor(der2l, dtype=dtype, device=device)
    der2u = torch.tensor(der2u, dtype=dtype, device=device)
    
    # der = der1 - -der2; % '-' as we calculate f(x) - p(x)
    der = torch.max(torch.abs(torch.stack([
        der1l - -der2l,
        der1l - -der2u,
        der1u - -der2l,
        der1u - -der2u
    ])))
    
    # determine number of points to sample
    dx = tol / der.item()
    reqPoints = int(torch.ceil((u_val - l_val) / dx).item() if isinstance(u_val - l_val, torch.Tensor) else np.ceil((u_val - l_val) / dx))
    numPoints = min(max(reqPoints, minPoints), maxPoints)
    
    # re-calculate tolerance with number of used points
    dx = (u_val - l_val) / numPoints
    tol = der.item() * dx
    
    # sample points - use torch
    x = torch.linspace(l_val, u_val, numPoints, dtype=dtype, device=device)
    x = torch.cat([l.unsqueeze(0) if l.numel() == 1 else l, x, u.unsqueeze(0) if u.numel() == 1 else u])  # add l, u
    diff = f(x) - _torch_polyval(coeffs, x)  # f works with torch, use torch polyval
    
    # find bounds
    diffl = torch.min(diff) - tol
    diffu = torch.max(diff) + tol
    
    return diffl, diffu

def _torch_polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate polynomial using torch (coeffs in descending order)"""
    result = torch.zeros_like(x)
    for i, c in enumerate(coeffs):
        result = result + c * (x ** (len(coeffs) - 1 - i))
    return result
