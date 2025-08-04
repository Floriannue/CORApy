"""
center - returns the center of a full-dimensional space; we define this
   to be the origin
   case R^0: 0 (not representable in MATLAB)

Syntax:
   c = center(fs)

Inputs:
   fs - fullspace object

Outputs:
   c - center

Example: 
   fs = fullspace(2);
   c = center(fs);

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

def center(fs):
    """
    Returns the center of a full-dimensional space; we define this
    to be the origin
    case R^0: 0 (not representable in MATLAB)
    
    Args:
        fs: fullspace object
        
    Returns:
        c: center
    """
    if fs.dimension == 0:
        c = np.nan
    else:
        c = np.zeros(fs.dimension)
    
    return c

# ------------------------------ END OF CODE ------------------------------ 