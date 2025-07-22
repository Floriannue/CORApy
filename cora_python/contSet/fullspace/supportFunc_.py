"""
supportFunc_ - calculates the upper or lower bound of a full-dimensional
   space along a certain direction
   case R^0: 'upper' -> +Inf, 'lower' -> -Inf

Syntax:
   val = supportFunc_(fs,dir,type)

Inputs:
   fs - fullspace object
   dir - direction for which the bounds are calculated (vector)
   type - upper bound, lower bound, or both ('upper','lower','range')

Outputs:
   val - bound of the full-dimensional space in the specified direction
   x - support vector

Example: 
   fs = fullspace(2);
   dir = [1;1];
   [val,x] = supportFunc(fs,dir);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/supportFunc

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   05-April-2023 (rename supportFunc_)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def supportFunc_(fs, dir_, type_='upper', *args):
    """
    Calculates the upper or lower bound of a full-dimensional
    space along a certain direction
    case R^0: 'upper' -> +Inf, 'lower' -> -Inf
    
    Args:
        fs: fullspace object
        dir_: direction for which the bounds are calculated (vector)
        type_: upper bound, lower bound, or both ('upper','lower','range')
        *args: additional arguments
        
    Returns:
        val: bound of the full-dimensional space in the specified direction
        x: support vector
    """
    dir_ = np.asarray(dir_).flatten()
    if fs.dimension == 0 and len(args) > 0:  # nargout == 2 equivalent
        raise CORAerror('CORA:notSupported',
                       'Intersection check of R^0 not supported')
    
    # set is always unbounded
    if type_ == 'upper':
        val = np.inf
        x = np.inf * np.ones(fs.dimension) * np.sign(dir_)
        # For zero directions, set x to 0 (not nan or inf)
        x[np.abs(dir_) == 0] = 0
    
    elif type_ == 'lower':
        val = -np.inf
        x = -np.inf * np.ones(fs.dimension) * np.sign(dir_)
        # For zero directions, set x to 0 (not nan or inf)
        x[np.abs(dir_) == 0] = 0
    
    elif type_ == 'range':
        # Import here to avoid circular imports
        from cora_python.contSet.interval import Interval
        val = Interval(-np.inf, np.inf)
        x = np.column_stack([-np.inf * np.ones(fs.dimension), 
                            np.inf * np.ones(fs.dimension)]) * np.sign(dir_)
        # For zero directions, set x to 0 (not nan or inf)
        x[np.abs(dir_) == 0, :] = 0
    
    # Always return both value and support vector
    return val, x

# ------------------------------ END OF CODE ------------------------------ 