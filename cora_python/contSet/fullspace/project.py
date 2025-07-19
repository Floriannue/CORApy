"""
project - projects a full-dimensional space onto the specified dimensions
   case R^0: no dimensions for projection possible

Syntax:
   fs = project(fs,dims)

Inputs:
   fs - fullspace object
   dims - dimensions for projection

Outputs:
   fs - projected fullspace

Example: 
   fs = fullspace(4);
   val = project(fs,1:2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def project(fs, dims):
    """
    Projects a full-dimensional space onto the specified dimensions
    case R^0: no dimensions for projection possible
    
    Args:
        fs: fullspace object
        dims: dimensions for projection
        
    Returns:
        fs: projected fullspace
    """
    if fs.dimension == 0:
        raise CORAerror('CORA:notSupported', 'Projection of of R^0 not supported')
    elif np.any(np.array(dims) < 0) or np.any(np.array(dims) > fs.dimension):
        raise CORAerror('CORA:outOfDomain', 'validDomain', f'1:{fs.dimension}')
    else:
        fs.dimension = len(dims)
    
    return fs

# ------------------------------ END OF CODE ------------------------------ 