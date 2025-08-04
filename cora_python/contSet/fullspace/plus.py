"""
plus - overloaded '+' operator for the Minkowski addition of a
   full-dimensional space and another set or vector
   case R^0: can only be added to another R^0, resulting in R^0;
             or to the empty set, resulting in the empty set

Syntax:
   S_out = fs + S
   S_out = plus(fs,S)

Inputs:
   fs - fullspace object, numeric
   S - contSet object, numeric

Outputs:
   res - fullspace object

Example: 
   fs = fullspace(2);
   Z = zonotope([1;1],[2 1; -1 0]);
   fs + Z

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
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric

def plus(fs, S):
    """
    Overloaded '+' operator for the Minkowski addition of a
    full-dimensional space and another set or vector
    case R^0: can only be added to another R^0, resulting in R^0;
              or to the empty set, resulting in the empty set
    
    Args:
        fs: fullspace object, numeric
        S: contSet object, numeric
        
    Returns:
        res: fullspace object
    """
    # call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(fs, 'precedence') and S.precedence < fs.precedence:
        S_out = S + fs
        return S_out
    
    # ensure that numeric is second input argument
    S_out, _ = reorderNumeric(fs, S)
    
    if S_out.dimension == 0:
        raise CORAerror('CORA:notSupported', 'Minkowski sum of R^0 not supported')
    
    # check dimensions of ambient space
    equal_dim_check(S_out, S)
    
    # empty set case
    if hasattr(S, 'representsa_') and callable(getattr(S, 'representsa_')):
        try:
            if S.representsa_('emptySet', 1e-15):
                # Import here to avoid circular imports
                from cora_python.contSet.emptySet import EmptySet
                S_out = EmptySet(fs.dim())
                return S_out
        except:
            pass
    
    # empty vector case (MATLAB: double.empty(n,0))
    if isinstance(S, np.ndarray) and S.size == 0:
        # Import here to avoid circular imports
        from cora_python.contSet.emptySet import EmptySet
        S_out = EmptySet(fs.dim())
        return S_out
    
    return S_out

# ------------------------------ END OF CODE ------------------------------ 