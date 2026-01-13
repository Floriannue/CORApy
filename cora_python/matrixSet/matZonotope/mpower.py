"""
mpower - Overloaded '^' operator for the power of matrix zonotope 

Syntax:
    matZ = mpower(matZ, exponent)
    matZ = matZ ** exponent

Inputs:
    matZ - matZonotope object
    exponent - exponent (nonnegative integer)

Outputs:
    matZ - matrix zonotope

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
Written:       18-June-2010 (MATLAB)
Last update:   05-August-2010 (MATLAB)
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def mpower(matZ: 'matZonotope', exponent: int) -> 'matZonotope':
    """
    Overloaded '^' operator for the power of matrix zonotope
    
    Args:
        matZ: matZonotope object
        exponent: exponent (nonnegative integer)
        
    Returns:
        matZpower: matrix zonotope raised to the power
        
    Raises:
        CORAerror: If exponent is not a nonnegative integer
    """
    # Check input arguments
    # MATLAB: inputArgsCheck({{matZ,'att','matZonotope'}, ...
    #                         {exponent,'att','numeric',{'scalar','integer','nonnegative'}}});
    inputArgsCheck([[matZ, 'att', 'matZonotope'],
                    [exponent, 'att', 'numeric', 'scalar', 'integer', 'nonnegative']])
    
    from .matZonotope import matZonotope
    from .dim import dim
    
    # MATLAB: if exponent==0
    if exponent == 0:
        # MATLAB: matZpower.C=eye(dim(matZ));
        # MATLAB: matZpower.G=zeros([size(matZ.C),0]);
        # Return identity matrix
        n = dim(matZ)[0]
        return matZonotope(np.eye(n), np.zeros((*matZ.C.shape, 0)))
    # MATLAB: elseif exponent==1
    elif exponent == 1:
        # MATLAB: do nothing
        return matZonotope(matZ.C, matZ.G)
    else:
        # MATLAB: matZpower=matZ*matZ;
        matZpower = matZ * matZ
        # MATLAB: for i=3:exponent
        for i in range(3, exponent + 1):  # Python range is exclusive, MATLAB is inclusive
            # MATLAB: matZpower=matZpower*matZ;
            matZpower = matZpower * matZ
        return matZpower
