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
    
    # Debug: Check eigenvalues and matrix norms
    if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
        try:
            eigenvals = np.linalg.eigvals(A)
            eigenvals_abs = np.abs(eigenvals)
            eigenvals_max = np.max(eigenvals_abs) if eigenvals.size > 0 else 0
            A_norm = np.linalg.norm(A, ord='fro')
            eAabst_max = np.max(np.abs(eAabst))
            Vdyn_center = Vdyn.center() if hasattr(Vdyn, 'center') else None
            Vdyn_center_max = np.max(np.abs(Vdyn_center)) if Vdyn_center is not None else 0
            Vdyn_generators = Vdyn.generators() if hasattr(Vdyn, 'generators') else None
            Vdyn_radius = np.sum(np.abs(Vdyn_generators), axis=1) if Vdyn_generators is not None else None
            Vdyn_radius_max = np.max(Vdyn_radius) if Vdyn_radius is not None and Vdyn_radius.size > 0 else 0
            print(f"[errorSolution_adaptive] Step {options['i']}: A max eigenval = {eigenvals_max:.6e}, A norm = {A_norm:.6e}, "
                  f"expm(A_abs*dt) max = {eAabst_max:.6e}, Vdyn center max = {Vdyn_center_max:.6e}, Vdyn radius max = {Vdyn_radius_max:.6e}, dt = {deltat:.6e}", flush=True)
        except Exception as e:
            print(f"[errorSolution_adaptive] Step {options['i']}: Error checking eigenvalues: {e}", flush=True)

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
    # Debug: Check initial AVsum
    if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
        try:
            AVsum_center = AVsum.center() if hasattr(AVsum, 'center') else None
            AVsum_generators = AVsum.generators() if hasattr(AVsum, 'generators') else None
            if AVsum_center is not None:
                AVsum_center_max = np.max(np.abs(AVsum_center))
                if AVsum_generators is not None:
                    AVsum_radius = np.sum(np.abs(AVsum_generators), axis=1)
                    AVsum_radius_max = np.max(AVsum_radius) if AVsum_radius.size > 0 else 0
                    print(f"[errorSolution_adaptive] Step {options['i']}: Initial AVsum (deltat*Vdyn): center max = {AVsum_center_max:.6e}, radius max = {AVsum_radius_max:.6e}, deltat = {deltat:.6e}", flush=True)
        except Exception as e:
            print(f"[errorSolution_adaptive] Step {options['i']}: Error checking initial AVsum: {e}", flush=True)
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
        # Debug: Check for infinite values in Taylor terms
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340) and eta <= 10:
            try:
                ApowerV_center = ApowerV.center() if hasattr(ApowerV, 'center') else None
                ApowerV_generators = ApowerV.generators() if hasattr(ApowerV, 'generators') else None
                if ApowerV_center is not None:
                    ApowerV_center_max = np.max(np.abs(ApowerV_center))
                    if ApowerV_generators is not None:
                        ApowerV_radius = np.sum(np.abs(ApowerV_generators), axis=1)
                        ApowerV_radius_max = np.max(ApowerV_radius) if ApowerV_radius.size > 0 else 0
                        temp_norm = np.linalg.norm(temp, ord='fro')
                        temp_max = np.max(np.abs(temp)) if temp.size > 0 else 0
                        print(f"[errorSolution_adaptive] Step {options['i']}, eta={eta}: ApowerV center max = {ApowerV_center_max:.6e}, ApowerV radius max = {ApowerV_radius_max:.6e}, temp norm = {temp_norm:.6e}, temp max = {temp_max:.6e}", flush=True)
                    elif np.any(np.isinf(ApowerV_center)) or ApowerV_center_max > 1e+50:
                        temp_norm = np.linalg.norm(temp, ord='fro')
                        print(f"[errorSolution_adaptive] Step {options['i']}, eta={eta}: ApowerV center max = {ApowerV_center_max:.6e}, temp norm = {temp_norm:.6e}", flush=True)
            except Exception:
                pass
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
                
                # Debug: Check components of Rerror
                if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
                    try:
                        AVsum_center = AVsum.center() if hasattr(AVsum, 'center') else None
                        AVsum_generators = AVsum.generators() if hasattr(AVsum, 'generators') else None
                        F_center = F.center() if hasattr(F, 'center') else None
                        F_generators = F.generators() if hasattr(F, 'generators') else None
                        if AVsum_center is not None:
                            AVsum_center_max = np.max(np.abs(AVsum_center))
                            if AVsum_generators is not None:
                                AVsum_radius = np.sum(np.abs(AVsum_generators), axis=1)
                                AVsum_radius_max = np.max(AVsum_radius) if AVsum_radius.size > 0 else 0
                                print(f"[errorSolution_adaptive] Step {options['i']}, eta={eta}: AVsum center max = {AVsum_center_max:.6e}, AVsum radius max = {AVsum_radius_max:.6e}", flush=True)
                        if F_center is not None:
                            F_center_max = np.max(np.abs(F_center))
                            if F_generators is not None:
                                F_radius = np.sum(np.abs(F_generators), axis=1)
                                F_radius_max = np.max(F_radius) if F_radius.size > 0 else 0
                                print(f"[errorSolution_adaptive] Step {options['i']}, eta={eta}: F center max = {F_center_max:.6e}, F radius max = {F_radius_max:.6e}", flush=True)
                        W_max = np.max(W) if W.size > 0 else 0
                        print(f"[errorSolution_adaptive] Step {options['i']}, eta={eta}: W max = {W_max:.6e}, deltat = {deltat:.6e}", flush=True)
                    except Exception as e:
                        print(f"[errorSolution_adaptive] Step {options['i']}, eta={eta}: Error checking components: {e}", flush=True)
                
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
