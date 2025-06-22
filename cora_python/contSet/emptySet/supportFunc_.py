"""
supportFunc_ - calculates the upper or lower bound of an empty set along
    a certain direction

Syntax:
    val = supportFunc_(O,dir,type)
    [val,x] = supportFunc_(O,dir,type)

Inputs:
    O - emptySet object
    dir - direction for which the bounds are calculated (vector)
    type - upper bound, lower bound, or both ('upper','lower','range')

Outputs:
    val - bound of the full-dimensional space in the specified direction
    x - support vector

Example: 
    O = EmptySet(2)
    dir = np.array([[1], [1]])
    val, x = supportFunc_(O, dir)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/supportFunc

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Last update:   05-April-2023 (rename supportFunc_) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from .emptySet import EmptySet


def supportFunc_(O: 'EmptySet', dir: np.ndarray, type_: str = 'upper', *varargin) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculates the upper or lower bound of an empty set along a certain direction
    
    Args:
        O: emptySet object
        dir: direction for which the bounds are calculated (vector)
        type_: upper bound, lower bound, or both ('upper','lower','range')
        *varargin: additional arguments (unused)
        
    Returns:
        val: bound of the full-dimensional space in the specified direction
        x: support vector (when called with two outputs)
    """
    # bounds are fixed by theory
    if type_ == 'upper':
        val = -np.inf
        x = np.empty((O.dimension, 0))

    elif type_ == 'lower':
        val = np.inf
        x = np.empty((O.dimension, 0))

    elif type_ == 'range':
        # Import interval here to avoid circular imports
        from cora_python.contSet.interval.interval import Interval
        val = Interval(-np.inf, np.inf)
        # actually, there would have to be two empty n-dimensional vectors next
        # to each other, but this is not supported...
        x = np.empty((O.dimension, 0))
    
    else:
        raise ValueError(f"Unknown type: {type_}. Must be 'upper', 'lower', or 'range'")
    
    # Return based on calling context (Python doesn't have MATLAB's nargout)
    # For now, always return both values
    return val, x 