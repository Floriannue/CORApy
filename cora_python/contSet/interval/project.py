"""
project - projects an interval onto the specified dimensions

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-September-2019 (MATLAB)
Last update: 21-May-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def project(I: 'Interval', dims) -> 'Interval':
    """
    Projects an interval onto the specified dimensions
    
    Args:
        I: Interval object
        dims: dimensions for projection (list or array of indices)
        
    Returns:
        Interval: projected interval
        
    Example:
        >>> I = Interval([-3, -5, -2], [3, 2, 1])
        >>> dims = [0, 2]  # Project onto 1st and 3rd dimensions
        >>> I_proj = project(I, dims)
    """
    from .interval import Interval
    
    # Check if interval is a matrix (not supported)
    if I.inf.ndim > 1 and I.inf.shape[0] > 1 and I.inf.shape[1] > 1:
        raise CORAError("CORA:wrongValue", 
                       "project not implemented for interval matrices")
    
    # Convert dims to numpy array for indexing
    dims = np.array(dims)
    
    # Handle scalar intervals (0-dimensional)
    if I.inf.ndim == 0:
        # For scalar intervals, only dimension 0 is valid
        if len(dims) == 1 and dims[0] == 0:
            inf_proj = I.inf
            sup_proj = I.sup
        else:
            raise CORAError("CORA:wrongInput", 
                           f"Invalid dimension {dims} for scalar interval")
    else:
        # Project bounds for multi-dimensional intervals
        inf_proj = I.inf[dims]
        sup_proj = I.sup[dims]
    
    return Interval(inf_proj, sup_proj)
