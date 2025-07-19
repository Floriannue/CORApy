"""
interval - converts a full-dimensional space to an interval object
   case R^0: since the only contained point 0 is not representable in
   MATLAB, we cannot convert R^0 to an interval

Syntax:
   I = interval(fs)

Inputs:
   fs - fullspace object

Outputs:
   I - interval

Example: 
   fs = fullspace(2);
   I = interval(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval/Inf

Authors:       Mark Wetzlinger, Tobias Ladner
Written:       22-March-2023
Last update:   25-February-2025 (TL, used polytope.Inf)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def interval(fs):
    """
    Converts a full-dimensional space to an interval object
    case R^0: since the only contained point 0 is not representable in
    MATLAB, we cannot convert R^0 to an interval
    
    Args:
        fs: fullspace object
        
    Returns:
        I: interval
    """
    if fs.dimension > 0:
        # lower and upper bounds are infinity
        # Import here to avoid circular imports
        from cora_python.contSet.interval import Interval
        I = Interval.Inf(fs.dimension)
    else:
        raise CORAerror('CORA:outOfDomain', 'validDomain', '>0')
    
    return I

# ------------------------------ END OF CODE ------------------------------ 