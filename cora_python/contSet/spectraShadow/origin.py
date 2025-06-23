"""
origin - instantiates a spectrahedral shadow representing the origin in R^n

Syntax:
    sS = origin(n)

Inputs:
    n - dimension

Outputs:
    sS - spectraShadow object representing the origin
"""

import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


def origin(n: int) -> SpectraShadow:
    """Instantiates a spectrahedral shadow representing the origin in R^n"""
    
    A = np.zeros((0, 0))
    c = np.zeros((n, 1))
    G = np.zeros((n, 0))
    
    return SpectraShadow(A, c, G) 