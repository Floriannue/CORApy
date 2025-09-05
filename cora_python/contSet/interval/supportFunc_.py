"""
supportFunc_ method for interval class
"""

import numpy as np
from typing import Union, Tuple
from .interval import Interval

def supportFunc_(I: Interval,
                 direction: np.ndarray,
                 type_: str = 'upper',
                 *args, **kwargs) -> Union[float, Tuple[float, np.ndarray]]:
    """
    supportFunc_ - calculates the upper or lower bound of an interval along a
    certain direction

    Syntax:
        val = supportFunc_(I,dir)
        [val,x] = supportFunc_(I,dir,type)

    Inputs:
        I - interval object
        direction - direction for which the bounds are calculated (vector)
        type_ - upper bound, lower bound, or both ('upper','lower','range')

    Outputs:
        val - bound of the interval in the specified direction
        x - support vector

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: contSet/supportFunc

    Authors:       Mark Wetzlinger
    Written:       27-March-2023
    Last update:   ---
    Last revision: ---
    Automatic python translation: Florian Nüssel BA 2025
    """
    
    # Ensure direction is a column vector
    direction = np.asarray(direction)
    if direction.ndim == 1:
        direction = direction.reshape(-1, 1)
    
    # Calculate bounds
    if type_ == 'upper':
        val = float(np.sum(direction * I.sup))  # Convert to scalar
        x = I.sup
    elif type_ == 'lower':
        val = float(np.sum(direction * I.inf))  # Convert to scalar
        x = I.inf
    elif type_ == 'range':
        lower_val = float(np.sum(direction * I.inf))  # Convert to scalar
        upper_val = float(np.sum(direction * I.sup))  # Convert to scalar
        val = Interval(lower_val, upper_val)
        x = np.column_stack([I.inf, I.sup])
    else:
        raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")
    
    return val, x 