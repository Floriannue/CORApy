"""
convHull_ - computes the convex hull of an empty set and another set or a
   point

Syntax:
   S_out = convHull_(O,S)

Inputs:
   O - emptySet object
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
"""

# Import required functions
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check

def convHull_(self, S=None, *args):
    """
    Computes the convex hull of an empty set and another set or a point
    
    Args:
        S: contSet object or numeric (optional)
        *args: additional arguments
        
    Returns:
        S_out: convex hull
    """
    # empty set is convex - if no second argument provided
    if S is None:
        return
    
    # ensure that numeric is second input argument
    O, S = reorder_numeric(self, S)
    
    # check dimensions
    equal_dim_check(O, S)
    
    # call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(O, 'precedence') and S.precedence < O.precedence:
        return S.convHull(O, *args)
    
    # convex hull is always the other set
    if hasattr(S, 'copy'):
        return S.copy()
    else:
        return S 