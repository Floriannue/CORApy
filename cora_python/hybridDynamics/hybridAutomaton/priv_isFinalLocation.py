"""
priv_isFinalLocation - checks if given location is final location

TRANSLATED FROM: cora_matlab/hybridDynamics/@hybridAutomaton/private/priv_isFinalLocation.m

Syntax:
    res = priv_isFinalLocation(loc,finalLoc)

Inputs:
    loc - number of current location (0-based in Python)
    finalLoc - vector/list of final locations (0-based in Python)

Outputs:
    res - true/false

Example:
    -

Authors:       Mark Wetzlinger (MATLAB)
Written:       --- (MATLAB)
Last update:   16-June-2022 (MW, simplify entire function, MATLAB)
               15-October-2024 (MW, rename, MATLAB)
Python translation: 2025
"""

from typing import Union, List
import numpy as np


def priv_isFinalLocation(loc: int, finalLoc: Union[np.ndarray, List[int]]) -> bool:
    """
    Checks if given location is final location.
    
    Args:
        loc: number of current location (0-based in Python)
        finalLoc: vector/list of final locations (0-based in Python)
                 Note: if no final location specified, finalLoc = 0 which results in false
                 as loc is always greater than or equal to zero
    
    Returns:
        bool: True if loc is in finalLoc, False otherwise
    """
    # MATLAB: res = any(loc == finalLoc);
    # Check if current location is equal to any of the possible final locations
    # note: if no final location specified, finalLoc = 0 which results in false
    # as loc is always greater than zero (in MATLAB, but >= 0 in Python)
    if isinstance(finalLoc, np.ndarray):
        return bool(np.any(loc == finalLoc))
    else:
        return loc in finalLoc

