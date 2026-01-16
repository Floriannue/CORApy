"""
errorSolution_adaptive - computes the solution due to the abstraction error
   the number of Taylor terms is chosen according to the set size decrease

Syntax:
    Rerror = errorSolution_adaptive(linsys,options,Vdyn,Vstat)

Inputs:
    linsys - linearSys object
    options - options struct (for nonlinear system)
    Vdyn - set of dynamic errors
    Vstat - set of static errors

Outputs:
    Rerror - reachable set due to the linearization error
    options - options struct (for nonlinear system)

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       24-April-2020 (MATLAB)
Last update:   27-May-2020
               15-June-2020 (include Vstat)
               10-July-2020 (delete Vstat from convergence process)
Python translation: 2025
"""

import math
from typing import Any, Dict, Tuple
import numpy as np
from scipy.linalg import expm

from cora_python.contSet.interval import Interval
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.representsa_ import representsa_


def errorSolution_adaptive(linsys: Any, options: Dict[str, Any], Vdyn: Any, Vstat: Any = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Computes the reachable set due to the abstraction error with adaptive Taylor terms.
    """
    # check if static error given
    # MATLAB: if nargin < 4 || representsa_(Vstat,'emptySet',1e-10)
    isVstat = not (Vstat is None or representsa_(Vstat, 'emptySet', 1e-10))

    # time step
    # MATLAB: deltat = options.timeStep;
    deltat = options['timeStep']

    # exponential
    # MATLAB: A = linsys.A;
    A = linsys.A
    A_abs = np.abs(A)
    Apower = [A]
    Apower_abs = [A_abs]
    M = np.eye(linsys.nr_of_dims)
    eAabst = expm(A_abs * deltat)

    # initialize Asum / AVsum
    AVsum = deltat * Vdyn
    RerrorInt_etanoF = _sum_abs_generators(AVsum)
    if isVstat:
        Asum = deltat * np.eye(linsys.nr_of_dims)

    eta = 1
    breakcond = False

    # loop over increasing Taylor terms
    while True:
        # compute powers
        # MATLAB: M = M + Apower_abs{eta} * deltat^eta / factorial(eta);
        M = M + Apower_abs[eta - 1] * deltat**eta / math.factorial(eta)

        # MATLAB: temp = deltat^(eta+1)/factorial(eta+1) * Apower{eta};
        temp = deltat**(eta + 1) / math.factorial(eta + 1) * Apower[eta - 1]
        ApowerV = temp * Vdyn
        AVsum = AVsum + ApowerV

        if isVstat:
            Asum = Asum + temp

        RerrorInt_etanoF = RerrorInt_etanoF + _sum_abs_generators(ApowerV)

        # at least two sets for comparison needed
        if eta > 1:
            gainnoF = 1 - formerEdgenoF / RerrorInt_etanoF[critDimnoF]

            # break if gain too small (or same truncation order as in previous
            # iteration of the same time step reached)
            if 'tt_err' in options and len(options['tt_err']) == options['i']:
                breakcond = eta == options['tt_err'][options['i'] - 1]
            elif gainnoF < options['zetaTabs'] or formerEdgenoF == 0:
                breakcond = True

            if breakcond:
                # determine error due to finite Taylor series
                W = np.abs(eAabst - M)
                # Interval matrix centered at 0 with width W
                E = IntervalMatrix(np.zeros_like(W), W)
                # error due to finite Taylor series
                F = (E * Vdyn) * deltat

                if isVstat:
                    # also former Asum due to E
                    eAtInt = Asum + (E * deltat)
                    Rerror = AVsum + F + (eAtInt * Vstat)
                else:
                    Rerror = AVsum + F

                # save taylor order for analysis
                if 'tt_err' not in options:
                    options['tt_err'] = []
                # MATLAB indexing is 1-based; options.i is 1-based, store at i-1
                if len(options['tt_err']) < options['i']:
                    options['tt_err'].append(eta)
                else:
                    options['tt_err'][options['i'] - 1] = eta
                break

            formerEdgenoF = RerrorInt_etanoF[critDimnoF]
        else:
            critDimnoF = int(np.argmax(RerrorInt_etanoF))
            formerEdgenoF = RerrorInt_etanoF[critDimnoF]

        # compute powers
        Apower.append(Apower[eta - 1] @ A)
        Apower_abs.append(Apower_abs[eta - 1] @ A_abs)
        eta += 1

    return Rerror, options


def _sum_abs_generators(Z: Any) -> np.ndarray:
    if hasattr(Z, 'generators') and callable(getattr(Z, 'generators')):
        G = Z.generators()
    elif hasattr(Z, 'G'):
        G = Z.G
    else:
        return np.zeros((Z.dim(),))
    if G is None or np.size(G) == 0:
        return np.zeros((G.shape[0],))
    return np.sum(np.abs(G), axis=1)
