"""
priv_expmRemainder - Computation of remainder term of exponential matrix

Authors:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 08-July-2021 (MATLAB)
Last update: 22-March-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
import scipy.linalg
import math
from cora_python.contSet.interval import Interval

def priv_expmRemainder(linsys, timeStep: float, truncationOrder: int):
    """
    Computation of remainder term of exponential matrix
    """

    # Check if it has already been computed
    if hasattr(linsys.taylor, 'E') and timeStep in getattr(linsys.taylor, '_E_cache', {}):
        return linsys.taylor._E_cache[timeStep]

    # Set a maximum order in case truncation order is given as Inf (adaptive)
    truncationOrderInf = np.isinf(truncationOrder)
    if truncationOrderInf:
        truncationOrder = 75

    # Initialization for loop
    A_abs = np.abs(linsys.A)
    n = linsys.A.shape[0]
    M = np.eye(n)
    options = {'timeStep': timeStep, 'ithpower': 1}

    # Compute powers for each term and sum of these
    # MATLAB: uses getTaylor(linsys,'Apower_abs',options) and getTaylor(linsys,'dtoverfac',options)
    for eta in range(1, int(truncationOrder) + 1):
        options['ithpower'] = eta
        # MATLAB: Apower_abs_i = getTaylor(linsys,'Apower_abs',options);
        Apower_abs_i = linsys.getTaylor('Apower_abs', **options)
        # MATLAB: dtoverfac_i = getTaylor(linsys,'dtoverfac',options);
        dtoverfac_i = linsys.getTaylor('dtoverfac', **options)

        # Additional term
        M_add = Apower_abs_i * dtoverfac_i

        # Adaptive handling
        if truncationOrderInf and np.all(M_add <= np.finfo(float).eps * M):
            break

        M = M + M_add

    # Determine error due to finite Taylor series
    # (compute absolute value of W for numerical stability)
    W = np.abs(scipy.linalg.expm(A_abs * timeStep) - M)
    E = Interval(-W, W)

    # Save in taylorLinSys object
    if not hasattr(linsys.taylor, '_E_cache'):
        linsys.taylor._E_cache = {}
    linsys.taylor._E_cache[timeStep] = E

    return E 