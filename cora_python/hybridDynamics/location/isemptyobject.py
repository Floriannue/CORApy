"""
isemptyobject - checks if a location object is empty

Syntax:
    res = isemptyobject(loc)

Inputs:
    loc - location object

Outputs:
    res - true/false

Example: 
    res = isemptyobject(location())

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       16-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any
import numpy as np


def isemptyobject(loc: Any) -> bool:
    """
    Check if a location object is empty
    
    Args:
        loc: Location object
    
    Returns:
        bool: True if location is empty, False otherwise
    """
    # MATLAB: [r,c] = size(loc);
    # For now, handle single objects (not arrays)
    
    # MATLAB: res(i,j) = isemptyobject(loc(i,j).contDynamics);
    if hasattr(loc, 'contDynamics'):
        if loc.contDynamics is None:
            return True
        # Check if contDynamics is empty array
        if isinstance(loc.contDynamics, np.ndarray) and loc.contDynamics.size == 0:
            return True
        # Check if contDynamics has isemptyobject method
        if hasattr(loc.contDynamics, 'isemptyobject'):
            return loc.contDynamics.isemptyobject()
    
    return False

