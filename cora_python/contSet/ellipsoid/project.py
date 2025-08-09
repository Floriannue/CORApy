import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
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
    
    # Convert dims to numpy array if it's not already
    dims = np.asarray(dims)
    
    # Check input arguments first (like MATLAB)
    inputArgsCheck([
        [E, 'att', 'ellipsoid'],
        [dims, 'att', ['numeric', 'logical'], [['nonnan', 'vector', 'integer', 'nonnegative'], ['vector']]]
    ])
    
    # Additional bounds checking (like MATLAB's @(dims) dims <= dim(E))
    n = E.dim()
    if dims.size == 0:
        raise CORAerror('CORA:wrongValue', '2nd', 'Projection dimensions cannot be empty')
    if np.any(dims >= n) or np.any(dims < 0):
        raise CORAerror('CORA:wrongValue', '2nd', f"Projection dimensions must be in range [0, {n-1}]")
    
    # Handle different input types
    if dims.dtype == bool:
        # Logical indexing - convert to indices
        if dims.size != n:
            raise CORAerror('CORA:wrongValue', '2nd', f"Logical projection mask must have length {n}")
        dims = np.where(dims)[0]
    else:
        # Numeric indexing - ensure integer type
        dims = np.asarray(dims, dtype=int)
        
        # Handle range syntax [start, end] vs individual indices [dim1, dim2, dim3, ...]
        # MATLAB supports both: [1,3] means range 1:3, [1,3,5] means individual indices
        # Only treat as range if it's exactly 2 elements and the second is larger AND within bounds
        if len(dims) == 2 and dims[1] > dims[0]:
            # Check if this could be a valid range (end index within bounds)
            n = E.dim()
            if dims[1] < n:
                # Range syntax: [start, end] -> range(start, end+1) (inclusive)
                start_idx = dims[0]
                end_idx = dims[1] + 1  # Make end inclusive
                dims = np.arange(start_idx, end_idx)
            # else: treat as individual indices even if second element is larger
        # else: individual indices - use as-is
    

    
    # Handle empty ellipsoid case
    if E.isemptyobject():
        # Return empty ellipsoid with projected dimension
        return Ellipsoid.empty(len(dims))
    
    # Project set using identity matrix approach (matching MATLAB)
    I = np.eye(n)
    P = I[dims, :]
    return P @ E 