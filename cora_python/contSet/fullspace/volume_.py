"""
volume_ - returns the volume of a full-dimensional space
   case R^0: NaN

Syntax:
   val = volume_(fs)

Inputs:
   fs - fullspace object

Outputs:
   val - volume

Example: 
   fs = fullspace(2);
   val = volume(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/volume

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   05-April-2023 (MW, rename volume_)
               25-April-2023 (MW, add R^0 case)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np

def volume_(fs, *args):
    """
    Returns the volume of a full-dimensional space
    case R^0: NaN
    
    Args:
        fs: fullspace object
        *args: additional arguments (unused)
        
    Returns:
        val: volume
    """
    if fs.dimension == 0:
        val = np.nan
    else:
        val = np.inf
    
    return val

# ------------------------------ END OF CODE ------------------------------ 