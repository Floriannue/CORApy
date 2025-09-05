"""
isequal - checks if a full-dimensional space is equal to another set or
   point; case R^0: can only be equal to another R^0

Syntax:
   res = isequal(fs,S)
   res = isequal(fs,S,tol)

Inputs:
   fs - fullspace object
   S - contSet object or numerical vector
   tol - (optional) tolerance

Outputs:
   res - true/false

Example: 
   fs1 = fullspace(2);
   fs2 = fullspace(3);
   res1 = isequal(fs1,fs1);
   res2 = isequal(fs1,fs2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def isequal(fs, S, *args):
    """
    Checks if a full-dimensional space is equal to another set or point;
    case R^0: can only be equal to another R^0
    
    Args:
        fs: fullspace object
        S: contSet object or numerical vector
        *args: additional arguments (tolerance)
        
    Returns:
        res: true/false
    """
    # Check number of arguments
    if len(args) > 1:
        raise ValueError("Too many arguments")
    
    # default values
    tol_values = set_default_values([np.finfo(float).eps], args)
    tol = tol_values[0]
    
    # check input arguments
    inputArgsCheck([[fs, 'att', ['fullspace', 'numeric']],
                   [S, 'att', ['contSet', 'numeric']],
                   [tol, 'att', 'numeric', ['scalar', 'nonnegative', 'nonnan']]])
    
    # ensure that numeric is second input argument
    fs, S = reorderNumeric(fs, S)
    
    # call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(fs, 'precedence') and S.precedence < fs.precedence:
        res = S.isequal(fs, tol)
        return res
    
    # ambient dimensions must match
    if not equal_dim_check(fs, S, True):
        res = False
        return res
    
    # Check if S is a fullspace
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Fullspace':
        res = fs.dimension == S.dimension
        return res
    
    # Check if S is an interval
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Interval':
        # only set that can cover R^n
        res = np.all(S.inf == -np.inf) and np.all(S.sup == np.inf)
        return res
    
    # no other sets can cover R^n or represent R^0
    if hasattr(S, '__class__') or isinstance(S, (int, float, np.ndarray)):
        res = False
        return res
    
    raise CORAerror('CORA:noops', fs, S)

# ------------------------------ END OF CODE ------------------------------ 