"""
dim - returns the dimension of a Taylor model

Syntax:
    n = dim(t)

Inputs:
    t - taylm object

Outputs:
    n - dimension
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.taylm.taylm import Taylm


def dim(t: 'Taylm') -> int:
    """Returns the dimension of a Taylor model"""
    
    if hasattr(t, 'monomials') and t.monomials.size > 0:
        return t.monomials.shape[1]
    elif hasattr(t, 'coefficient') and hasattr(t.coefficient, '__len__'):
        return len(t.coefficient)
    else:
        return 0 