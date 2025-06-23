"""
dim - returns the dimension of the ambient space of a spectrahedral shadow

Syntax:
    n = dim(sS)

Inputs:
    sS - spectraShadow object

Outputs:
    n - dimension of the ambient space

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   05-April-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spectraShadow import SpectraShadow


def dim(sS: 'SpectraShadow') -> int:
    """
    Returns the dimension of the ambient space of a spectrahedral shadow
    
    Args:
        sS: spectraShadow object
        
    Returns:
        n: dimension of the ambient space
    """
    
    if hasattr(sS, 'c') and sS.c.size > 0:
        if sS.c.ndim == 1:
            return len(sS.c)
        else:
            return sS.c.shape[0]
    elif hasattr(sS, 'G') and sS.G.size > 0:
        return sS.G.shape[0]
    else:
        return 0 