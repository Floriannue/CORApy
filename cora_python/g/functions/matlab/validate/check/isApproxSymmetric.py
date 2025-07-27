import numpy as np
from typing import Union
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def isApproxSymmetric(Q: np.ndarray, TOL: float) -> bool:
    """
    isApproxSymmetric - Checks if a shape matrix is symmetric (within
    tolerance)

    Syntax:
        res = isApproxSymmetric(Q,TOL)

    Inputs:
        Q - square matrix
        TOL - tolerance

    Outputs:
        res - true/false indicating whether Q is symmetric (within tolerance)

    Example: 
        # E = ellipsoid([1,0;0,1/2],[1;1]);
        # res = isApproxSymmetric(E.Q,E.TOL);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Victor Gassmann
    Written:       13-March-2019
    Last update:   15-October-2019
    Last revision: ---
    Automatic python translation: Florian Nüssel BA 2025
    """

    # take default value for tolerance if none given
    # In MATLAB, TOL could be omitted and it would use ellipsoid.empty(1).TOL.
    # In Python, we will assume TOL is always provided as per function signature
    # or handle defaults at a higher level (e.g., in the ellipsoid constructor itself).

    # The MATLAB code uses triu(Q) and tril(Q)' for comparison.
    # In numpy, we can achieve this with np.triu and np.tril and then transposing.
    # However, for symmetry check, a simpler approach is to directly compare Q with its transpose.
    # If a matrix is symmetric, Q == Q.T. We check this within tolerance.
    return np.all(withinTol(Q, Q.T, TOL)) 