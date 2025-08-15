"""
lift_ - Lift a polytope to a higher-dimensional space with unbounded new dimensions

Mirrors MATLAB @polytope/lift_.m:
- N: target ambient dimension
- dims: positions in the higher-dimensional space corresponding to original coordinates
- New (non-listed) dimensions are unbounded (no constraints added for them)

Always uses H-representation since V-representation cannot encode Inf properly.
"""

import numpy as np
from typing import Sequence, TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def lift_(P: 'Polytope', N: int, dims: Sequence[int]) -> 'Polytope':
    from .polytope import Polytope

    if N < P.dim():
        raise CORAerror('CORA:wrongValue', 'second',
                        'Dimension of higher-dimensional space must be larger than or equal to the dimension of the given polytope.')
    if len(dims) != P.dim():
        raise CORAerror('CORA:wrongValue', 'third',
                        'Number of dimensions in higher-dimensional space must match the dimension of the given polytope.')

    # Ensure constraints available
    P.constraints()

    # Project constraints to higher-dimensional space: place original columns at dims indices
    A_high = np.zeros((P.A.shape[0], N)) if P.A.size > 0 else np.zeros((0, N))
    if P.A.size > 0:
        A_high[:, np.array(dims) - 1] = P.A
    Ae_high = np.zeros((P.Ae.shape[0], N)) if P.Ae.size > 0 else np.zeros((0, N))
    if P.Ae.size > 0:
        Ae_high[:, np.array(dims) - 1] = P.Ae

    P_out = Polytope(A_high, P.b, Ae_high, P.be)

    return P_out


