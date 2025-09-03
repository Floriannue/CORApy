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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def empty(n: int = 0) -> Taylm:
    """Instantiates an empty Taylor model"""
    
    # Validate input
    if n < 0:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Dimension must be non-negative')
    
    # Create empty Taylor model
    monomials = np.zeros((0, n)) if n > 0 else np.zeros((0, 0))
    coefficients = np.array([])
    remainder = np.array([0, 0])  # Empty interval
    
    return Taylm(monomials, coefficients, remainder) 