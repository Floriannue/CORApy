"""
contains_ - determines if a full-dimensional space contains a set or a
   point
   case R^0: only contains R^0, 0 (not representable in MATLAB), and []

Syntax:
   res = contains_(fs,S)
   res = contains_(fs,S,type)
   res = contains_(fs,S,type,tol)

Inputs:
   fs - fullspace object
   S - contSet object or numerical vector
   method - method used for the containment check.
      Currently, the only available options are 'exact' and 'approx'.
   tol - tolerance for the containment check; the higher the
      tolerance, the more likely it is that points near the boundary of C
      will be detected as lying in C, which can be useful to counteract
      errors originating from floating point errors.
   maxEval - Currently has no effect
   certToggle - if set to 'true', cert will be computed (see below),
      otherwise cert will be set to NaN.
   scalingToggle - if set to 'true', scaling will be computed (see
      below), otherwise scaling will be set to inf.

Outputs:
   res - true/false
   cert - returns true iff the result of res could be
          verified. For example, if res=false and cert=true, S is
          guaranteed to not be contained in fs, whereas if res=false and
          cert=false, nothing can be deduced (S could still be
          contained in fs).
          If res=true, then cert=true.
   scaling - returns the smallest number 'scaling', such that
          scaling*(fs - fs.center) + fs.center contains S.

Example: 
   fs = fullspace(2);
   p = [1;1];
   res = contains(fs,p);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/contains

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   05-April-2023 (MW, rename contains_)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def contains_(fs, S, method=None, tol=None, maxEval=None, certToggle=None, scalingToggle=None, *args):
    """
    Determines if a full-dimensional space contains a set or a point
    case R^0: only contains R^0, 0 (not representable in MATLAB), and []
    
    Args:
        fs: fullspace object
        S: contSet object or numerical vector
        method: method used for the containment check
        tol: tolerance for the containment check
        maxEval: Currently has no effect
        certToggle: if set to 'true', cert will be computed
        scalingToggle: if set to 'true', scaling will be computed
        *args: additional arguments
        
    Returns:
        res: true/false
        cert: returns true iff the result of res could be verified
        scaling: returns the smallest number 'scaling'
    """
    if fs.dimension == 0:
        raise CORAerror('CORA:notSupported',
                       'Containment check for R^0 not supported')
    
    # full-dimensional space contains all other sets, including itself
    res = True
    cert = True
    scaling = 0
    
    return res, cert, scaling

# ------------------------------ END OF CODE ------------------------------ 