"""
empty - instantiates an empty Taylor model

Syntax:
    t = empty(n)

Inputs:
    n - dimension

Outputs:
    t - empty taylm object
"""

import numpy as np
from cora_python.contSet.taylm.taylm import Taylm


def empty(n: int = 0) -> Taylm:
    """Instantiates an empty Taylor model"""
    
    # Create empty Taylor model
    monomials = np.zeros((0, n)) if n > 0 else np.zeros((0, 0))
    coefficients = np.array([])
    remainder = np.array([0, 0])  # Empty interval
    
    return Taylm(monomials, coefficients, remainder) 