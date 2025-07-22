"""
norm_ - compute the maximum Euclidean norm of an ellipsoid

Syntax:
    val = norm_(E)
    val = norm_(E, type)

Inputs:
    E    - ellipsoid object
    type - (optional) norm type (default: 2)

Outputs:
    val - value of the maximum norm

Example:
    E = Ellipsoid(np.array([[3, -1], [-1, 1]]), np.zeros((2, 1)))
    val = E.norm_()

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/norm

Authors:       Victor Gassmann (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       20-November-2019 (MATLAB)
Last update:   27-March-2023 (MW, rename norm_, MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def norm_(E, type=2, *args, **kwargs):
    """
    Compute the maximum Euclidean norm of an ellipsoid.
    Args:
        E: ellipsoid object
        type: norm type (default: 2)
    Returns:
        val: value of the maximum norm
    """
    if not (isinstance(type, (int, float)) and type == 2):
        raise CORAerror('CORA:noExactAlg', E, type, 'Only implemented for Euclidean norm')

    # Handle empty ellipsoid (Q is empty)
    if E.Q.size == 0:
        return -np.inf

    # Only for zero-centers implemented
    if not np.allclose(E.q, 0):
        raise CORAerror('CORA:notSupported', 'Not yet implemented for non-zero center.')

    # Transform into eigenspace
    lmax = np.max(np.linalg.eigvalsh(E.Q))
    val = np.sqrt(lmax)

    # Check for empty set
    if np.isnan(val) and E.representsa_('emptySet', 0):
        val = -np.inf
    return val 