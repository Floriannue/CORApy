"""
project - project an interval to a lower-dimensional subspace

Syntax:
    I_proj = project(I, dims)

Inputs:
    I - interval object
    dims - dimensions for projection (list or array of integers)

Outputs:
    I_proj - projected interval object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List


def project(I, dims: Union[List[int], np.ndarray]):
    """
    Project interval to lower-dimensional subspace
    
    Args:
        I: interval object
        dims: dimensions for projection (1-based indexing)
        
    Returns:
        Projected interval object
    """
    # Import here to avoid circular imports
    from .interval import interval
    
    # Handle empty interval
    if I.inf.size == 0:
        return interval.empty(len(dims))
    
    # Convert dims to 0-based indexing and ensure it's a list
    if isinstance(dims, (int, np.integer)):
        dims = [dims]
    dims = [d - 1 for d in dims]  # Convert to 0-based indexing
    
    # Check dimension validity
    n_dims = I.dim()
    if isinstance(n_dims, int):
        max_dim = n_dims
    else:
        max_dim = max(n_dims) if isinstance(n_dims, list) else 1
    
    for d in dims:
        if d < 0 or d >= max_dim:
            raise ValueError(f"Dimension {d+1} is out of range for {max_dim}-dimensional interval")
    
    # Project bounds
    if I.inf.ndim == 0:
        # Scalar interval - can only project to dimension 1
        if dims != [0]:
            raise ValueError(f"Cannot project scalar interval to dimensions {[d+1 for d in dims]}")
        inf_proj = I.inf
        sup_proj = I.sup
    elif I.inf.ndim == 1:
        # Vector interval
        inf_proj = I.inf[dims]
        sup_proj = I.sup[dims]
    else:
        # Matrix interval - project rows
        inf_proj = I.inf[dims, :]
        sup_proj = I.sup[dims, :]
    
    return interval(inf_proj, sup_proj) 