import numpy as np
from .interval import Interval

def cat(dim, *varargin):
    """
    cat - Overloaded 'cat()' operator for intervals, concatenates 
       given intervals along the given dimension
    
    Syntax:
       I = cat(dim,I1,I2,...)
    
    Inputs:
       dim - dimension to concatenate along (1-based for MATLAB compatibility)
       I1,I2,... - interval object
    
    Outputs:
       I - interval object
    """
    if not all(isinstance(i, Interval) for i in varargin):
        # Allow concatenation with numeric types by converting them
        try:
            intervals = [i if isinstance(i, Interval) else Interval(i) for i in varargin]
        except Exception:
            raise TypeError("All arguments to be concatenated must be Interval objects or convertible to them.")
    else:
        intervals = varargin

    # Python uses 0-based indexing for axis
    axis = dim - 1
    
    # Pre-process arrays to handle 1D cases consistently
    processed_infs = []
    processed_sups = []
    for i in intervals:
        inf = i.inf
        sup = i.sup
        # Promote 1D arrays to 2D row vectors for consistent concatenation
        if inf.ndim == 1:
            inf = inf.reshape(1, -1)
            sup = sup.reshape(1, -1)
        processed_infs.append(inf)
        processed_sups.append(sup)
    
    # Concatenate using numpy
    try:
        inf_cat = np.concatenate(processed_infs, axis=axis)
        sup_cat = np.concatenate(processed_sups, axis=axis)
    except ValueError as e:
        # Provide a more informative error
        raise ValueError(f"Failed to concatenate intervals along axis {axis}. Check dimensions.") from e
        
    return Interval(inf_cat, sup_cat) 