"""
projectHighDim_ - projects a full-dimensional space onto a higher-dimensional space
   case R^0: undefined

Syntax:
   fs = projectHighDim_(fs,N,dims)

Inputs:
   fs - fullspace object
   N - dimension of the higher-dimensional space
   proj - states of the high-dimensional space that correspond to the
         states of the low-dimensional space

Outputs:
   fs - projected fullspace

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/projectHighDim, contSet/lift

Authors:       Tobias Ladner
Written:       19-September-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def projectHighDim_(fs, N, proj):
    """
    Projects a full-dimensional space onto a higher-dimensional space
    case R^0: undefined
    
    Args:
        fs: fullspace object
        N: dimension of the higher-dimensional space
        proj: states of the high-dimensional space that correspond to the
              states of the low-dimensional space
        
    Returns:
        fs: projected fullspace
    """
    if len(proj) == N:
        # keep as is
        pass
    else:
        # projection to higher dimension is not defined as function expects new
        # dimensions to be bounded at 0
        raise CORAerror('CORA:notDefined', 'Cannot bound new dimensions at 0', 'contSet/lift')
    
    return fs

# ------------------------------ END OF CODE ------------------------------ 