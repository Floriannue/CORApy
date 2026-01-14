"""
priv_correctionMatrixState - Compute correction matrix for the state

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
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .priv_expmRemainder import priv_expmRemainder

def priv_correctionMatrixState(linsys, timeStep: float, truncationOrder: float):
    """
    Compute correction matrix for the state, see [1, Prop. 3.1]

    References:
        [1] M. Althoff. "Reachability Analysis and its Application to the
            Safety Assessment of Autonomous Cars", PhD Dissertation, 2010.
    """

    # Check if it has already been computed
    if hasattr(linsys.taylor, 'F') and timeStep in getattr(linsys.taylor, '_F_cache', {}):
        return linsys.taylor._F_cache[timeStep]

    # Set a maximum order in case truncation order is given as Inf (adaptive)
    truncationOrderInf = np.isinf(truncationOrder)
    if truncationOrderInf:
        truncationOrder = 75

    n = linsys.A.shape[0]
    Asum_pos_F = np.zeros((n, n))
    Asum_neg_F = np.zeros((n, n))

    for eta in range(2, int(truncationOrder) + 1):
        # Compute factor
        exp1 = -eta / (eta - 1)
        exp2 = -1 / (eta - 1)
        dtoverfac = (timeStep ** eta) / math.factorial(eta)
        factor = (eta ** exp1 - eta ** exp2) * dtoverfac

        # Get positive and negative parts of A^eta
        Apower = linsys.taylor.computeField('Apower', ithpower=eta)
        Asum_add_pos = np.maximum(Apower, 0)
        Asum_add_neg = np.minimum(Apower, 0)
        Asum_add_pos = factor * Asum_add_pos
        Asum_add_neg = factor * Asum_add_neg

        # Break condition in case truncation order is selected adaptively
        if truncationOrderInf:
            if (np.all(np.abs(Asum_add_neg) <= np.finfo(float).eps * np.abs(Asum_pos_F)) and
                np.all(np.abs(Asum_add_pos) <= np.finfo(float).eps * np.abs(Asum_neg_F))):
                break
            elif eta == truncationOrder:
                raise CORAerror('CORA:notConverged', 'Time step size too big for computation of F.')

        # Compute powers; factor is always negative
        Asum_pos_F = Asum_pos_F + Asum_add_neg
        Asum_neg_F = Asum_neg_F + Asum_add_pos

    # Compute correction matrix for the state
    # MATLAB: F = interval(Asum_neg_F,Asum_pos_F);
    # F is a 2D interval matrix, so use IntervalMatrix
    F = IntervalMatrix(Asum_neg_F, Asum_pos_F)

    # Compute/read remainder of exponential matrix (unless truncationOrder=Inf)
    if not truncationOrderInf:
        E = priv_expmRemainder(linsys, timeStep, truncationOrder)
        # E is an Interval, but F is an IntervalMatrix, so convert E to IntervalMatrix
        if isinstance(E, Interval):
            # E is a 2D interval (from matrix W), convert to IntervalMatrix
            E_matrix = IntervalMatrix(E.inf, E.sup)
        else:
            E_matrix = E
        F = F + E_matrix

    # Save in taylorLinSys object
    if not hasattr(linsys.taylor, '_F_cache'):
        linsys.taylor._F_cache = {}
    linsys.taylor._F_cache[timeStep] = F

    return F 