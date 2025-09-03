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
    
    # Handle empty Taylor model - check if monomials array exists but is empty
    if hasattr(t, 'monomials'):
        if isinstance(t.monomials, np.ndarray):
            if len(t.monomials.shape) == 2 and t.monomials.shape[1] > 0:
                # For 2D arrays, dimension is the second axis (number of variables)
                # This works even for empty arrays with 0 rows but >0 columns
                return t.monomials.shape[1]
            elif len(t.monomials.shape) == 1 and t.monomials.shape[0] > 0:
                # For 1D arrays, dimension is the length
                return t.monomials.shape[0]
            elif t.monomials.size == 0:
                # Only return 0 if the array is truly empty (no structure)
                return 0
        elif isinstance(t.monomials, list) and len(t.monomials) == 0:
            return 0
    
    # For Taylor models, the dimension is the number of variables
    # This can be inferred from the monomials structure
    if hasattr(t, 'monomials') and len(t.monomials) > 0:
        # Each monomial should have the same number of variables
        if isinstance(t.monomials, list):
            return t.monomials[0].shape[0] if hasattr(t.monomials[0], 'shape') else len(t.monomials[0])
        else:
            return t.monomials.shape[1]  # For numpy arrays, dimension is the second axis
    elif hasattr(t, 'names_of_var') and t.names_of_var:
        return len(t.names_of_var)
    else:
        return 1  # Default to 1D 