"""
polytope - converts a full-dimensional space to a polytope object
   case R^0: since the only contained point 0 is not representable in
   MATLAB, we cannot convert R^0 to a polytope

Syntax:
   P = polytope(fs)

Inputs:
   fs - fullspace object

Outputs:
   P - polytope object

Example: 
   fs = fullspace(2);
   P = polytope(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polytope/Inf

Authors:       Mark Wetzlinger, Tobias Ladner
Written:       14-December-2023
Last update:   25-February-2025 (TL, used polytope.Inf)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def polytope(fs):
    """
    Converts a full-dimensional space to a polytope object
    case R^0: since the only contained point 0 is not representable in
    MATLAB, we cannot convert R^0 to a polytope
    
    Args:
        fs: fullspace object
        
    Returns:
        P: polytope object
    """
    if fs.dimension > 0:
        # init polytope/inf with correct dimensions
        # Import here to avoid circular imports
        from cora_python.contSet.polytope import Polytope
        P = Polytope.Inf(fs.dimension)
    else:
        raise CORAerror('CORA:outOfDomain', 'validDomain', '>0')
    
    return P

# ------------------------------ END OF CODE ------------------------------ 