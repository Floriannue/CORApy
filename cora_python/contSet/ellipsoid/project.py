import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def project(E: 'Ellipsoid', dims: np.ndarray) -> 'Ellipsoid':
    """
    project - projects an ellipsoid onto the specified dimensions

    Syntax:
        E = project(E,dims)

    Inputs:
        E - (ellipsoid) ellipsoid
        dims - dimensions for projection

    Outputs:
        E - (ellipsoid) projected ellipsoid

    Example: 
        E = ellipsoid([9.3 -0.6 1.9;-0.6 4.7 2.5; 1.9 2.5 4.2])
        E = project(E,[1 3])

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Victor Gassmann (MATLAB)
                   Python translation by AI Assistant
    Written:       13-March-2019 (MATLAB)
    Last update:   04-July-2022 (VG, input checks, MATLAB)
    Python translation: 2025
    """
    from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
    
    # Convert dims to numpy array if it's not already
    dims = np.asarray(dims)
    
    # Handle different input types
    if dims.dtype == bool:
        # Logical indexing - convert to indices
        dims = np.where(dims)[0]
    else:
        # Numeric indexing - ensure integer type
        dims = np.asarray(dims, dtype=int)
        
        # Handle range syntax [start, end] vs individual indices [dim1, dim2, dim3, ...]
        if len(dims) == 2 and dims[1] > dims[0]:
            # Range syntax: [start, end] -> range(start, end+1) (inclusive)
            start_idx = dims[0]
            end_idx = dims[1] + 1  # Make end inclusive
            dims = np.arange(start_idx, end_idx)
        # else: individual indices - use as-is
    
    # Check input arguments - all indices should be 0-based
    inputArgsCheck([
        [E, 'att', 'ellipsoid'],
        [dims, 'att', ['numeric', 'logical'], [['nonnan', 'vector', 'integer', 'nonnegative'], ['vector']]]
    ])
    
    # Validate dimensions are within bounds
    n = E.dim()
    if np.any(dims >= n) or np.any(dims < 0):
        raise ValueError(f"Projection dimensions must be in range [0, {n-1}]")
    
    # Project set using identity matrix approach (matching MATLAB)
    I = np.eye(n)
    P = I[dims, :]
    return P @ E 