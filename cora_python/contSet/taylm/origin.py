"""
origin - instantiates a Taylor model representing the origin

Syntax:
    t = origin(n)

Inputs:
    n - dimension

Outputs:
    t - taylm object representing the origin
"""

import numpy as np
from cora_python.contSet.taylm.taylm import Taylm


def origin(n: int) -> Taylm:
    """Instantiates a Taylor model representing the origin"""
    
    # Create constant zero Taylor model
    monomials = np.zeros((1, n))  # Constant term
    coefficients = np.array([0])
    remainder = np.array([0, 0])  # Zero interval
    
    return Taylm(monomials, coefficients, remainder) 