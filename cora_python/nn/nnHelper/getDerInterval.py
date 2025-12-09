"""
getDerInterval - compute bounds for the derivative of a polynomial

Syntax:
    [derl,deru] = nnHelper.getDerInterval(coeffs, l, u)

Inputs:
    coeffs - coefficients of polynomial
    l - lower bound of input domain
    u - upper bound of input domain

Outputs:
    [derl,deru] - interval bounding the derivative

Other m-files required: none
Subfunctions: fpolyder
MAT-files required: none

See also: fpolyder

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Union
from cora_python.g.functions.matlab.polynomial.fpolyder import fpolyder


def getDerInterval(coeffs: Union[np.ndarray, torch.Tensor], l: Union[float, np.ndarray, torch.Tensor], 
                   u: Union[float, np.ndarray, torch.Tensor]) -> tuple[float, float]:
    """
    Compute the maximum and the minimum derivative of the given polynomial.
    Internal to nn - works with torch tensors.
    
    Args:
        coeffs: coefficients of polynomial (torch tensor expected internally)
        l: lower bound of input domain (torch tensor or float expected internally)
        u: upper bound of input domain (torch tensor or float expected internally)
        
    Returns:
        derl, deru: interval bounding the derivative within [l,u] (floats)
        
    See also: -
    """
    # Convert to numpy for fpolyder and np.roots (external functions that use numpy)
    if isinstance(coeffs, torch.Tensor):
        coeffs_np = coeffs.cpu().numpy()
    else:
        coeffs_np = coeffs
    
    if isinstance(l, torch.Tensor):
        l_val = l.item() if l.numel() == 1 else float(l.cpu().numpy())
    else:
        l_val = float(l)
    
    if isinstance(u, torch.Tensor):
        u_val = u.item() if u.numel() == 1 else float(u.cpu().numpy())
    else:
        u_val = float(u)
    
    # find extreme points of derivative of polynomial
    p = coeffs_np
    dp = fpolyder(p)  # fpolyder uses numpy
    dp2 = fpolyder(dp)
    
    # Find roots of second derivative (critical points of first derivative)
    # np.roots requires numpy
    dp2_roots = np.roots(dp2)
    # Filter imaginary roots (keep only real roots)
    dp2_roots = dp2_roots[np.imag(dp2_roots) == 0]
    
    # evaluate extreme points of derivative
    points = np.concatenate([[l_val], dp2_roots, [u_val]])
    # Filter points within bounds [l, u]
    points = points[(l_val <= points) & (points <= u_val)]
    
    # Evaluate derivative at these points
    dp_y = np.polyval(dp, points)
    
    derl = float(np.min(dp_y))
    deru = float(np.max(dp_y))
    
    return derl, deru
