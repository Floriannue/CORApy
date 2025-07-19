"""
isIntersecting_ - checks if the dimension of the affine hull of a
   full-dimensional space is equal to the dimension of its ambient space
   case R^0: only point in vector space is 0 (not representable in
   MATLAB), so isIntersecting would always return true

Syntax:
   res = isIntersecting_(fs,S,type,tol)

Inputs:
   fs - fullspace object
   S - contSet object or numerical vector
   type - type of check
   tol - tolerance

Outputs:
   res - true/false

Example: 
   fs = fullspace(2);
   p = [1;1]
   res = isIntersecting(fs,p);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/isIntersecting

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   05-April-2023 (MW, rename isIntersecting_)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def isIntersecting_(fs, S, type=None, tol=None, *args):
    """
    Checks if the dimension of the affine hull of a
    full-dimensional space is equal to the dimension of its ambient space
    case R^0: only point in vector space is 0 (not representable in
    MATLAB), so isIntersecting would always return true
    
    Args:
        fs: fullspace object
        S: contSet object or numerical vector
        type: type of check
        tol: tolerance
        *args: additional arguments
        
    Returns:
        res: true/false
    """
    # ensure that numeric is second input argument
    fs, S = reorderNumeric(fs, S)
    
    if fs.dimension == 0:
        raise CORAerror('CORA:notSupported',
                       'Intersection check of R^0 not supported')
    
    # all singletons intersect the full space
    if isinstance(S, (int, float, list, tuple)) or hasattr(S, 'dtype'):
        res = True
        return res
    
    # intersection with empty set is empty
    if hasattr(S, '__class__') and S.__class__.__name__ == 'EmptySet':
        res = False
        return res
    
    # other set must not be empty
    if hasattr(S, 'representsa_') and callable(getattr(S, 'representsa_')):
        res = not S.representsa_('emptySet', tol)
        return res
    
    raise CORAerror('CORA:noops', fs, S)

# ------------------------------ END OF CODE ------------------------------ 