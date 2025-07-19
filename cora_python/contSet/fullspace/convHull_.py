"""
convHull_ - computes the convex hull of a fullspace and another set or a
   point

Syntax:
   S_out = convHull_(fs,S)

Inputs:
   fs - fullspace object
   S - contSet object or numeric

Outputs:
   S_out - convex hull

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/convHull

Authors:       Mark Wetzlinger
Written:       29-September-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric

def convHull_(fs, S, *args):
    """
    Computes the convex hull of a fullspace and another set or a point
    
    Args:
        fs: fullspace object
        S: contSet object or numeric
        *args: additional arguments
        
    Returns:
        S_out: convex hull
    """
    # full space is convex
    if S is None:
        return fs
    
    # ensure that numeric is second input argument
    fs, S = reorderNumeric(fs, S)
    
    # check dimensions
    equal_dim_check(fs, S)
    
    # call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(fs, 'precedence') and S.precedence < fs.precedence:
        S_out = S.convHull(fs, *args)
        return S_out
    
    # convex hull is full space
    S_out = fs
    
    return S_out

# ------------------------------ END OF CODE ------------------------------ 