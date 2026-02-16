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
    
    # REMOVED: Debug code that computes expensive operations even when not printing
    # Only compute debug info if actually needed (when progress is True AND we're printing)

    # initialize Asum / AVsum
    # Check if Vdyn contains infinite values before using it
    try:
        Vdyn_center = Vdyn.center() if hasattr(Vdyn, 'center') else None
        if Vdyn_center is not None and np.any(np.isinf(Vdyn_center)):
            from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
            raise CORAerror('CORA:reachSetExplosion', 'VerrorDyn contains infinite values.')
    except Exception:
        pass  # If check fails, continue (might not have center method)
    
    AVsum = deltat * Vdyn
    # REMOVED: Debug code that computes expensive operations even when not printing
    RerrorInt_etanoF = np.asarray(_sum_abs_generators(AVsum), dtype=np.float64)
    if isVstat:
        Asum = deltat * np.eye(linsys.nr_of_dims)

    eta = 1
    breakcond = False
    
    # Optimize factorial computation: compute incrementally instead of calling math.factorial
    # factorial(eta+1) = factorial(eta) * (eta+1)
    fact_eta = 1.0  # factorial(1) = 1
    fact_eta_plus_1 = 2.0  # factorial(2) = 2

    # loop over increasing Taylor terms
    while True:
        # compute powers
        # MATLAB: M = M + Apower_abs{eta} * deltat^eta / factorial(eta);
        M = M + Apower_abs[eta - 1] * deltat**eta / fact_eta

        # MATLAB: temp = deltat^(eta+1)/factorial(eta+1) * Apower{eta};
        temp = deltat**(eta + 1) / fact_eta_plus_1 * Apower[eta - 1]
        ApowerV = temp * Vdyn
        # REMOVED: Debug code that computes expensive operations even when not printing
        AVsum = AVsum + ApowerV

        if isVstat:
            Asum = Asum + temp

        RerrorInt_etanoF = RerrorInt_etanoF + np.asarray(_sum_abs_generators(ApowerV), dtype=np.float64)

        # at least two sets for comparison needed
        if eta > 1:
            # MATLAB: gainnoF = 1 - formerEdgenoF / RerrorInt_etanoF(critDimnoF). Use float64 for exact match.
            gainnoF = float(1.0 - float(formerEdgenoF) / float(RerrorInt_etanoF[critDimnoF]))

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
                
                # REMOVED: Debug code that computes expensive operations even when not printing
                
                # Check if Rerror contains infinite values
                try:
                    Rerror_center = Rerror.center() if hasattr(Rerror, 'center') else None
                    if Rerror_center is not None and np.any(np.isinf(Rerror_center)):
                        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                        raise CORAerror('CORA:reachSetExplosion', 'Rerror contains infinite values.')
                except Exception:
                    pass  # If check fails, continue

                # save taylor order for analysis
                if 'tt_err' not in options:
                    options['tt_err'] = []
                # MATLAB indexing is 1-based; options.i is 1-based, store at i-1
                if len(options['tt_err']) < options['i']:
                    options['tt_err'].append(eta)
                else:
                    options['tt_err'][options['i'] - 1] = eta
                break

            formerEdgenoF = float(RerrorInt_etanoF[critDimnoF])
        else:
            critDimnoF = int(np.argmax(RerrorInt_etanoF))
            formerEdgenoF = float(RerrorInt_etanoF[critDimnoF])

        # compute powers
        Apower.append(Apower[eta - 1] @ A)
        Apower_abs.append(Apower_abs[eta - 1] @ A_abs)
        eta += 1
        
        # Update factorials incrementally (much faster than math.factorial)
        # fact_eta_plus_1 is now fact_eta for next iteration
        fact_eta = fact_eta_plus_1
        fact_eta_plus_1 = fact_eta * (eta + 1)  # factorial(eta+1) = factorial(eta) * (eta+1)

    return Rerror, options


def _sum_abs_generators(Z: Any) -> np.ndarray:
    if hasattr(Z, 'generators') and callable(getattr(Z, 'generators')):
        G = Z.generators()
    elif hasattr(Z, 'G'):
        G = Z.G
    else:
        return np.zeros((Z.dim(),), dtype=np.float64)
    if G is None or np.size(G) == 0:
        return np.zeros((G.shape[0],), dtype=np.float64)
    return np.asarray(np.sum(np.abs(G), axis=1), dtype=np.float64)
