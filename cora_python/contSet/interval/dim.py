"""
dim - dimension of an interval

Syntax:
    n = dim(I)

Inputs:
    I - interval object

Outputs:
    n - dimension of the interval

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval


def dim(self: 'Interval') -> Union[int, List[int]]:
    """
    Returns the dimension of the ambient space of an interval.
    This method overrides the default ContSet dim method.
    """
    if not hasattr(self, 'inf'):
        # This case can happen if the object is not fully initialized
        return 0

    infi = self.inf

    # Handle empty interval: shape is (n, 0)
    if infi.ndim == 2 and infi.shape[1] == 0:
        return infi.shape[0]

    shape = np.shape(infi)

    # In numpy, scalars have shape (), vectors have (n,), matrices (n,m)
    # MATLAB's size() is different. A scalar is 1x1, a vector is nx1 or 1xn.

    # Scalar case
    if shape == ():
        return 1

    # Vector case
    if len(shape) == 1:
        return shape[0]

    # Matrix case
    if len(shape) == 2:
        rows, cols = shape
        # In MATLAB, a 1-D vector can be n x 1 or 1 x n.
        # In numpy, it is often (n,). But it can be (n,1) or (1,n)
        if rows == 1:
            return cols
        elif cols == 1:
            return rows
        else:  # This is a matrix interval
            return list(shape)

    # Higher-dimensional case
    return list(shape)
