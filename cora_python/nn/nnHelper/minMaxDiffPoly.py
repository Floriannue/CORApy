"""
minMaxDiffPoly - compute the maximum and the minimum difference between two polynomials

Syntax:
    [diffl,diffu] = nnHelper.minMaxDiffPoly(coeffs1, coeffs2, l, u)

Inputs:
    coeffs1 - coefficients of first polynomial
    coeffs2 - coefficients of second polynomial
    l - lower bound of input domain
    u - upper bound of input domain

Outputs:
    [diffl,diffu] - interval bounding the lower and upper error

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

def minMaxDiffPoly(coeffs1: Union[np.ndarray, torch.Tensor], coeffs2: Union[np.ndarray, torch.Tensor], 
                   l: Union[float, np.ndarray, torch.Tensor], u: Union[float, np.ndarray, torch.Tensor]) -> tuple:
    """
    Compute the maximum and the minimum difference between two polynomials.
    Internal to nn - works with torch tensors.
    
    Args:
        coeffs1: coefficients of first polynomial (ascending order like MATLAB) (torch tensor expected internally)
        coeffs2: coefficients of second polynomial (ascending order like MATLAB) (torch tensor expected internally)
        l: lower bound of input domain (torch tensor or float expected internally)
        u: upper bound of input domain (torch tensor or float expected internally)
        
    Returns:
        Tuple of (diffl, diffu) interval bounding the lower and upper error (floats)
    """
    # Convert to numpy for fpolyder and np.roots (external functions that use numpy)
    if isinstance(coeffs1, torch.Tensor):
        coeffs1_np = coeffs1.cpu().numpy()
    else:
        coeffs1_np = coeffs1
    
    if isinstance(coeffs2, torch.Tensor):
        coeffs2_np = coeffs2.cpu().numpy()
    else:
        coeffs2_np = coeffs2
    
    if isinstance(l, torch.Tensor):
        l_val = l.item() if l.numel() == 1 else float(l.cpu().numpy())
    else:
        l_val = float(l)
    
    if isinstance(u, torch.Tensor):
        u_val = u.item() if u.numel() == 1 else float(u.cpu().numpy())
    else:
        u_val = float(u)
    
    # MATLAB style: pad to same length, subtract
    # MATLAB: p = zeros(1,max(length(coeffs1),length(coeffs2)));
    #         p(end-length(coeffs1)+1:end) = coeffs1;
    #         p(end-length(coeffs2)+1:end) = p(end-length(coeffs2)+1:end)-coeffs2;
    max_len = max(len(coeffs1_np), len(coeffs2_np))
    p = np.zeros(max_len)
    p[-len(coeffs1_np):] = coeffs1_np
    p[-len(coeffs2_np):] = p[-len(coeffs2_np):] - coeffs2_np
    
    # compute derivative
    dp = fpolyder(p)  # fpolyder uses numpy
    
    # find roots of derivative
    roots = np.roots(dp)  # np.roots requires numpy
    
    # filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # filter roots within domain
    real_roots = real_roots[(real_roots > l_val) & (real_roots < u_val)]
    
    # add boundary points
    points = np.concatenate([[l_val], real_roots, [u_val]])
    
    # evaluate difference at all points
    # p is in ascending order (like MATLAB), and np.polyval handles this correctly
    values = np.polyval(p, points)
    
    # find bounds
    diffl = float(np.min(values))
    diffu = float(np.max(values))
    
    return diffl, diffu
