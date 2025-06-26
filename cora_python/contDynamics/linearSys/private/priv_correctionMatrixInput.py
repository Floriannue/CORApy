"""
priv_correctionMatrixInput - Compute correction matrix for the input

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
import math
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .priv_expmRemainder import priv_expmRemainder

def priv_correctionMatrixInput(linsys, timeStep: float, truncationOrder: float):
    """
    Compute correction matrix for the input, see [1, p. 38]

    References:
        [1] M. Althoff. "Reachability Analysis and its Application to the
            Safety Assessment of Autonomous Cars", PhD Dissertation, 2010.
    """

    # Check if it has already been computed
    if hasattr(linsys.taylor, 'G') and timeStep in getattr(linsys.taylor, '_G_cache', {}):
        return linsys.taylor._G_cache[timeStep]

    # Set a maximum order in case truncation order is given as Inf (adaptive)
    truncationOrderInf = np.isinf(truncationOrder)
    if truncationOrderInf:
        truncationOrder = 75

    n = linsys.A.shape[0]
    Asum_pos_G = np.zeros((n, n))
    Asum_neg_G = np.zeros((n, n))

    for eta in range(2, int(truncationOrder) + 2):
        # Compute factor
        exp1 = -eta / (eta - 1)
        exp2 = -1 / (eta - 1)
        dtoverfac = (timeStep ** eta) / math.factorial(eta)
        factor = (eta ** exp1 - eta ** exp2) * dtoverfac

        # Get positive and negative parts of A^(eta-1)
        Apower = linsys.taylor.computeField('Apower', ithpower=eta - 1)
        Asum_add_pos = np.maximum(Apower, 0)
        Asum_add_neg = np.minimum(Apower, 0)
        Asum_add_pos = factor * Asum_add_pos
        Asum_add_neg = factor * Asum_add_neg

        # Compute ratio for floating-point precision
        if truncationOrderInf:
            if (np.all(np.abs(Asum_add_neg) <= np.finfo(float).eps * np.abs(Asum_pos_G)) and
                np.all(np.abs(Asum_add_pos) <= np.finfo(float).eps * np.abs(Asum_neg_G))):
                break
            elif eta == truncationOrder + 1:
                raise CORAerror('CORA:notConverged', 'Time step size too big for computation of G.')

        # Compute powers; factor is always negative
        Asum_pos_G = Asum_pos_G + Asum_add_neg
        Asum_neg_G = Asum_neg_G + Asum_add_pos

    # Compute correction matrix for input
    G = Interval(Asum_neg_G, Asum_pos_G)

    # Compute/read remainder of exponential matrix (unless truncationOrder=Inf)
    if not truncationOrderInf:
        E = priv_expmRemainder(linsys, timeStep, truncationOrder)
        G = G + E * timeStep

    # Save in taylorLinSys object
    if not hasattr(linsys.taylor, '_G_cache'):
        linsys.taylor._G_cache = {}
    linsys.taylor._G_cache[timeStep] = G

    return G 