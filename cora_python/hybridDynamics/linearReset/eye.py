"""
eye - instantiates an identity reset function of dimension n

TRANSLATED FROM: cora_matlab/hybridDynamics/@linearReset/eye.m

Syntax:
    linReset = LinearReset.eye(n)
    linReset = LinearReset.eye(n, m)

Inputs:
    n - pre/post-state dimension
    m - input dimension (optional, default: 1)

Outputs:
    linReset - linearReset object

Example: 
    n = 3; m = 2;
    linReset1 = LinearReset.eye(n);
    linReset2 = LinearReset.eye(n, m);

Authors:       Mark Wetzlinger (MATLAB)
Written:       15-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Optional
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    from .linearReset import LinearReset


def eye(n: int, m: Optional[int] = None) -> 'LinearReset':
    """
    Instantiates an identity reset function of dimension n.
    
    Args:
        n: pre/post-state dimension
        m: input dimension (optional, default: 1)
    
    Returns:
        LinearReset object with identity matrix A, zero B and c
    """
    from .linearReset import LinearReset
    
    # Set default value for input dimension
    if m is None:
        m = 1
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'integer', 'nonnegative']],
        [m, 'att', 'numeric', ['scalar', 'integer', 'nonnegative']]
    ])
    
    # Instantiate reset function
    if n == 0:
        linReset = LinearReset()
    else:
        linReset = LinearReset(np.eye(n), np.zeros((n, m)), np.zeros((n, 1)))
    
    return linReset
