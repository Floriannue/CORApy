"""
empty - instantiates an empty spectrahedral shadow

Syntax:
    sS = empty(n)

Inputs:
    n - dimension

Outputs:
    sS - empty spectraShadow object
"""

import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


def empty(n: int = 0) -> SpectraShadow:
    """Instantiates an empty spectrahedral shadow"""
    
    A = np.zeros((0, 0))
    c = np.zeros((n, 1)) if n > 0 else np.zeros((0, 1))
    G = np.zeros((n, 0)) if n > 0 else np.zeros((0, 0))
    
    return SpectraShadow(A, c, G) 