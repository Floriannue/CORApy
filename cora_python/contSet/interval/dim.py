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
    # Get infimum (equivalent to supremum for dimension purposes)
    infi = self.inf
    
    # Handle empty interval: shape is (n, 0)
    if infi.size == 0:
        return infi.shape[0] if len(infi.shape) > 0 else 0

    # Determine size
    dims = infi.shape

    if len(dims) <= 2:
        # 1-d or 2-d interval
        if len(dims) == 1:
            # Vector case: (n,) -> n
            return dims[0]
        else:
            # Matrix case: (rows, cols)
            rows, cols = dims
            if rows == 1:
                return cols
            elif cols == 1:
                return rows
            else:
                return list(dims)
    else:
        # n-d interval
        return list(dims)
