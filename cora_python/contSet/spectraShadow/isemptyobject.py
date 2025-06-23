"""
isemptyobject - checks whether a spectrahedral shadow contains any information

Syntax:
    res = isemptyobject(sS)

Inputs:
    sS - spectraShadow object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       07-June-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


def isemptyobject(sS: 'SpectraShadow') -> bool:
    """
    Checks whether a spectrahedral shadow contains any information
    
    Args:
        sS: spectraShadow object
        
    Returns:
        res: true if the spectrahedral shadow is empty, false otherwise
    """
    
    # Check if center is empty
    if not hasattr(sS, 'c') or sS.c.size == 0:
        return True
    
    # If center exists but has zero dimension
    if hasattr(sS, 'c') and sS.c.size > 0:
        return sS.dim() == 0
    
    return False 