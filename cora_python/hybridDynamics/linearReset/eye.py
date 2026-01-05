"""
eye - instantiates an identity reset function of dimension n

Syntax:
    linReset = linearReset.eye(n)
    linReset = linearReset.eye(n,m)

Inputs:
    n - pre/post-state dimension
    m - input dimension

Outputs:
    linReset - linearReset object

Example: 
    n = 3; m = 2;
    linReset1 = linearReset.eye(n);
    linReset2 = linearReset.eye(n,m);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       15-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Optional
import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.hybridDynamics.linearReset.linearReset import LinearReset


def eye(n: int, m: Optional[int] = None) -> LinearReset:
    """
    Instantiate an identity reset function of dimension n
    
    Args:
        n: Pre/post-state dimension
        m: Input dimension (default: 1)
    
    Returns:
        LinearReset: Identity reset function
    """
    # MATLAB: check number of input arguments and set default number of inputs
    # narginchk(1,2);
    # m = setDefaultValues({1},varargin);
    if m is None:
        m = 1
    else:
        # If m is provided, it should be in varargin format
        pass
    
    # MATLAB: check input arguments
    # inputArgsCheck({{n,'att','numeric',{'scalar','integer','nonnegative'}};
    #                 {m,'att','numeric',{'scalar','integer','nonnegative'}}});
    inputArgsCheck([
        [n, 'att', 'numeric', ('scalar', 'integer', 'nonnegative')],
        [m, 'att', 'numeric', ('scalar', 'integer', 'nonnegative')]
    ])
    
    # MATLAB: instantiate reset function
    # if n == 0
    #     linReset = linearReset();
    # else
    #     linReset = linearReset(eye(n),zeros(n,m),zeros(n,1));
    # end
    if n == 0:
        linReset = LinearReset()
    else:
        A = np.eye(n)
        B = np.zeros((n, m))
        c = np.zeros((n, 1))
        linReset = LinearReset(A, c, B)
    
    return linReset

