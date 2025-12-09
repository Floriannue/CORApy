"""
leastSquarePolyFunc - determine the optimal polynomial function fit that 
   minimizes the squared distance to the data points

Syntax:
    coeffs = nnHelper.leastSquarePolyFunc(x, y, n)

Inputs:
    x - x values
    y - y values
    n - polynomial order

Outputs:
    coeffs - coefficients of resulting polynomial

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       17-September-2021
Last update:   ---
Last revision: 28-March-2022 (TL)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Union

def leastSquarePolyFunc(x: Union[float, np.ndarray, torch.Tensor], y: Union[float, np.ndarray, torch.Tensor], n: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Determine the optimal polynomial function fit that minimizes the squared distance to the data points.
    Internal to nn - works with torch tensors.
    
    Args:
        x: x values (torch tensor expected internally)
        y: y values (torch tensor expected internally)
        n: polynomial order
        
    Returns:
        coeffs: coefficients of resulting polynomial (torch tensor)
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    device = x.device
    dtype = x.dtype
    
    # Match MATLAB exactly: A = x'.^(0:n) - use torch
    A = torch.stack([x ** i for i in range(n + 1)], dim=1)
    
    # Match MATLAB exactly: coeffs = pinv(A) @ y' - use torch
    # torch.pinverse for pseudo-inverse
    coeffs = torch.linalg.pinv(A) @ y
    
    # Match MATLAB exactly: coeffs = fliplr(coeffs') - use torch
    coeffs = torch.flip(coeffs, dims=[0])
    
    return coeffs
