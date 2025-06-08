"""
dim - dimension of an interval

Syntax:
    n = dim(I)

Inputs:
    I - interval object

Outputs:
    n - dimension of the interval

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def dim(obj) -> int:
    """
    Get dimension of the interval
    
    Args:
        obj: interval object
        
    Returns:
        Dimension of the interval
    """
    infi = obj.inf
    
    # For empty intervals, return the number of rows (first dimension)
    if infi.size == 0:
        return infi.shape[0] if infi.ndim > 0 else 0
    
    # Determine size following MATLAB logic
    dims = infi.shape
    
    # Handle scalar case
    if len(dims) == 0:
        return 1
    
    if len(dims) <= 2:
        # 1-d or 2-d interval
        rows = dims[0]
        cols = dims[1] if len(dims) > 1 else 1
        if rows == 1:
            return cols
        elif cols == 1:
            return rows
        else:
            return [rows, cols]  # Return list for matrix intervals
    else:
        # n-d interval
        return list(dims) 