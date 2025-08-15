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
    monomials = np.zeros((1, n))  # symbolic variable placeholders (unused)
    coefficients = np.zeros(1)
    from cora_python.contSet.interval.interval import Interval
    remainder = Interval(np.zeros((n, 1)), np.zeros((n, 1)))
    
    return Taylm(monomials, coefficients, remainder) 