"""
volume_ - Computes volume of an interval

Syntax:
    vol = volume_(I)

Inputs:
    I - interval object

Outputs:
    vol - volume

Example: 
    I = interval([1; -1], [2; 1])
    vol = volume(I)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/volume

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 24-July-2016 (MATLAB)
Last update: 18-August-2022 (MW, include standardized preprocessing) (MATLAB)
             04-December-2023 (MW, fix degenerate-unbounded case) (MATLAB)
Last revision: 27-March-2023 (MW, rename volume_) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def volume_(I: Interval, *args) -> float:
    """
    Computes volume of an interval
    
    Args:
        I: Interval object
        *args: Additional arguments (for compatibility)
        
    Returns:
        vol: Volume of the interval
    """
    # Compute half of the diameter
    r = I.rad()
    
    if r.size == 0 or not np.all(r):
        # empty or degenerate
        vol = 0.0
    else:
        # simple volume formula: product of all diameters
        vol = np.prod(2 * r)
    
    return float(vol) 