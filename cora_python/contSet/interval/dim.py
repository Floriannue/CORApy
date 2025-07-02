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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval


def dim(self: 'Interval') -> int:
    """
    Returns the dimension of the interval.
    
    This method overrides the default ContSet dim method.
    """
    if not hasattr(self, 'inf') or self.inf.size == 0:
        # Special case for empty intervals created with Interval.empty(dim)
        # These might have a dim property set before inf/sup are populated.
        if hasattr(self, '_dim'):
             return self._dim
        return 0
    
    if self.inf.ndim == 0:
        return 1  # Scalar
    
    # For vectors (1D arrays) or column/row vectors (2D arrays)
    if self.inf.ndim == 1:
        return len(self.inf)
    else:  # ndim >= 2
        return self.inf.shape[0]
