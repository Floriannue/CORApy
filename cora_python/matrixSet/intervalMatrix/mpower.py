"""
mpower - Overloaded '^' operator for the power of an interval matrix

Syntax:
    intMatPower = mpower(intMat, exponent)
    intMatPower = intMat ** exponent

Inputs:
    intMat - intervalMatrix object
    exponent - exponent (nonnegative integer)

Outputs:
    intMatPower - interval matrix 

Example: 
    M = IntervalMatrix(np.eye(2), 2*np.eye(2))
    M2 = M ** 2

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
Written:       21-June-2010 (MATLAB)
Last update:   05-August-2010 (MATLAB)
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def mpower(intMat: 'IntervalMatrix', exponent: int) -> 'IntervalMatrix':
    """
    Overloaded '^' operator for the power of an interval matrix
    
    Args:
        intMat: IntervalMatrix object
        exponent: exponent (nonnegative integer)
        
    Returns:
        intMatPower: interval matrix raised to the power
        
    Raises:
        CORAerror: If exponent is not a nonnegative integer
    """
    # Check input arguments
    # MATLAB: inputArgsCheck({{intMat,'att','intervalMatrix'}, ...
    #                         {exponent,'att','numeric',{'nonnegative','integer','scalar'}}});
    inputArgsCheck([[intMat, 'att', 'IntervalMatrix'],
                    [exponent, 'att', 'numeric', 'nonnegative', 'integer', 'scalar']])
    
    from .intervalMatrix import IntervalMatrix
    
    # MATLAB: if exponent==0
    if exponent == 0:
        # MATLAB: intMatPower.int=intMat.int^0;
        # Return identity matrix
        n = intMat.int.shape[0]
        return IntervalMatrix(np.eye(n), np.zeros((n, n)))
    # MATLAB: elseif exponent==1
    elif exponent == 1:
        # MATLAB: do nothing
        return intMat
    else:
        # MATLAB: intMatPower=intMat*intMat;
        intMatPower = intMat * intMat
        # MATLAB: for i=3:exponent
        for i in range(3, exponent + 1):  # Python range is exclusive, MATLAB is inclusive
            # MATLAB: intMatPower=intMatPower*intMat;
            intMatPower = intMatPower * intMat
        return intMatPower
