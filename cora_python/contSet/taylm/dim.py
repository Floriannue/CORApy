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
    
    # Handle empty Taylor model
    if hasattr(t, 'monomials') and len(t.monomials) == 0:
        return 0
    
    # For Taylor models, the dimension is the number of variables
    # This can be inferred from the monomials structure
    if hasattr(t, 'monomials') and len(t.monomials) > 0:
        # Each monomial should have the same number of variables
        return t.monomials[0].shape[0] if hasattr(t.monomials[0], 'shape') else len(t.monomials[0])
    elif hasattr(t, 'names_of_var') and t.names_of_var:
        return len(t.names_of_var)
    else:
        return 1  # Default to 1D 