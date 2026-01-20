"""
priv_abstractionError_adaptive - computes the abstraction error
note: no Taylor Model or zoo integration

Syntax:
    [VerrorDyn,VerrorStat,err,options] = priv_abstractionError_adaptive(nlnsys,options,R,Rdiff,...
     H,Zdelta,VerrorStat,T,ind3,Zdelta3)

Inputs:
    nlnsys - nonlinearSys object
    R - time-interval solution of current step (incl. adm. abstr. err)
    U - input set
    options - options struct
    Rdiff - set of state differences [2,(6)]
    remaining inputs: same as outputs of precompStatError (only 'poly')

Outputs:
    VerrorDyn - set based on abstraction error
    VerrorStat - set based on abstraction error
    err - over-approximated abstraction error
    options - options struct

References:
  [1] M. Althoff et al. "Reachability Analysis of Nonlinear Systems with 
      Uncertain Parameters using Conservative Linearization"
  [2] M. Althoff et al. "Reachability analysis of nonlinear systems using 
      conservative polynomialization and non-convex sets"

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       14-January-2020 (MATLAB)
Last update:   14-April-2020
               20-November-2023 (MW, store generators selected for reduction)
Python translation: 2025
"""

from typing import Any, Dict, Tuple, Optional, List
import numpy as np

from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def priv_abstractionError_adaptive(nlnsys: Any, R: Any, Rdiff: Any, U: Any, options: Dict[str, Any],
                                  H: Any = None, Zdelta: Any = None, VerrorStat: Any = None,
                                  T: Optional[Any] = None, ind3: Optional[list] = None, Zdelta3: Optional[Any] = None
                                  ) -> Tuple[Any, Any, np.ndarray, Dict[str, Any]]:
    """
    Compute abstraction error for adaptive nonlinear reachability.
    """
    # compute interval of reachable set and input (at origin)
    IH_x = R.interval()
    IH_u = U.interval()

    # translate intervals by linearization point (at traj / lin. point)
    totalInt_x = IH_x + nlnsys.linError.p.x
    totalInt_u = IH_u + nlnsys.linError.p.u

    # LIN; TENSOR 2 -----------------------------------------------------------
    if options['alg'] == 'lin' and options['tensorOrder'] == 2:
        # assign correct hessian (using interval arithmetic)
        nlnsys = nlnsys.setHessian('int')

        # obtain maximum absolute values within IH, IHinput
        IHinf = np.abs(IH_x.infimum())
        IHsup = np.abs(IH_x.supremum())
        dx = np.maximum(IHinf, IHsup)

        IHinputInf = np.abs(IH_u.infimum())
        IHinputSup = np.abs(IH_u.supremum())
        du = np.maximum(IHinputInf, IHinputSup)

        # compute an over-approximation of the Lagrange remainder [1,Prop.1]
        dz = np.vstack([dx, du])

        # evaluate the hessian matrix with interval arithmetic
        try:
            if not options.get('isHessianConst', False):
                H = nlnsys.hessian(totalInt_x, totalInt_u)
                # very first step: check if Hessian is constant
                if not options.get('hessianCheck', False):
                    options = _aux_checkIfHessianConst(nlnsys, options, H, totalInt_x, totalInt_u)
        except CORAerror as exc:
            if exc.identifier == 'interval:setoutofdomain':
                raise CORAerror('reach:setoutofdomain', 'Interval Arithmetic: Set out of domain.')
            raise

        err = np.zeros((nlnsys.nr_of_dims, 1))
        if not options.get('isHessianConst', False):
            for i in range(nlnsys.nr_of_dims):
                H_abs = abs(H[i])
                H_ = np.maximum(H_abs.infimum(), H_abs.supremum())
                # Debug: Check for infinite values in H_ or dz
                if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
                    H_max = np.max(np.abs(H_)) if H_.size > 0 else 0
                    dz_max = np.max(np.abs(dz)) if dz.size > 0 else 0
                    if H_max > 1e+50 or dz_max > 1e+50:
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}, dim {i}: H_max = {H_max:.6e}, dz_max = {dz_max:.6e}", flush=True)
                err_val = 0.5 * (dz.T @ H_ @ dz)
                # Check for infinite result
                if np.isinf(err_val) or np.isnan(err_val):
                    if options.get('progress', False):
                        print(f"[priv_abstractionError_adaptive] ERROR: err[{i}] = {err_val} (inf/nan) at step {options['i']}. "
                              f"H_max = {np.max(np.abs(H_)):.6e}, dz_max = {np.max(np.abs(dz)):.6e}", flush=True)
                    raise CORAerror('CORA:reachSetExplosion', f'Abstraction error computation produced {err_val} at dimension {i}.')
                err[i, 0] = err_val
        else:
            for i in range(nlnsys.nr_of_dims):
                err_val = 0.5 * (dz.T @ options['hessianConst'][i] @ dz)
                if np.isinf(err_val) or np.isnan(err_val):
                    if options.get('progress', False):
                        print(f"[priv_abstractionError_adaptive] ERROR: err[{i}] = {err_val} (inf/nan) with const Hessian at step {options['i']}", flush=True)
                    raise CORAerror('CORA:reachSetExplosion', f'Abstraction error computation produced {err_val} at dimension {i}.')
                err[i, 0] = err_val

        VerrorDyn = Zonotope(np.zeros_like(err), np.diag(err.flatten()))
        VerrorStat = []

    # LIN; TENSOR 3 -----------------------------------------------------------
    elif options['alg'] == 'lin' and options['tensorOrder'] == 3:
        nlnsys = nlnsys.setHessian('standard')
        nlnsys = nlnsys.setThirdOrderTensor('int')

        H = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u)
        dz = Interval.vertcat(IH_x, IH_u)

        # Debug: Check R size before reduction
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
            try:
                R_center = R.center() if hasattr(R, 'center') else None
                R_generators = R.generators() if hasattr(R, 'generators') else None
                if R_center is not None:
                    R_center_max = np.max(np.abs(R_center))
                    if R_generators is not None:
                        R_radius = np.sum(np.abs(R_generators), axis=1)
                        R_radius_max = np.max(R_radius) if R_radius.size > 0 else 0
                        R_num_generators = R_generators.shape[1] if R_generators.ndim > 1 else 0
                        R_generator_max = np.max(np.abs(R_generators)) if R_generators.size > 0 else 0
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}: R before reduce: center max = {R_center_max:.6e}, radius max = {R_radius_max:.6e}, num gens = {R_num_generators}, gen max = {R_generator_max:.6e}", flush=True)
                    else:
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}: R before reduce: center max = {R_center_max:.6e}", flush=True)
                # Check if R is PolyZonotope
                if hasattr(R, 'G') and hasattr(R, 'GI'):
                    R_G_max = np.max(np.abs(R.G)) if R.G.size > 0 else 0
                    R_GI_max = np.max(np.abs(R.GI)) if R.GI.size > 0 else 0
                    R_G_num = R.G.shape[1] if R.G.size > 0 else 0
                    R_GI_num = R.GI.shape[1] if R.GI.size > 0 else 0
                    print(f"[priv_abstractionError_adaptive] Step {options['i']}: R (PolyZonotope): G max = {R_G_max:.6e} ({R_G_num} gens), GI max = {R_GI_max:.6e} ({R_GI_num} gens)", flush=True)
            except Exception as e:
                print(f"[priv_abstractionError_adaptive] Step {options['i']}: Error checking R: {e}", flush=True)
        
        # reduce zonotope
        if 'gredIdx' in options and len(options['gredIdx'].get('Rred', [])) == options['i']:
            Rred = R.reduce('idx', options['gredIdx']['Rred'][options['i'] - 1])
        else:
            Rred_res = R.reduce('adaptive', np.sqrt(options['redFactor']))
            if isinstance(Rred_res, tuple):
                Rred, _, idx = Rred_res
                if 'gredIdx' in options:
                    options['gredIdx'].setdefault('Rred', []).append(idx)
            else:
                Rred = Rred_res

        # Debug: Check Rred size before cartProd_
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
            try:
                Rred_center = Rred.center() if hasattr(Rred, 'center') else None
                Rred_generators = Rred.generators() if hasattr(Rred, 'generators') else None
                if Rred_center is not None:
                    Rred_center_max = np.max(np.abs(Rred_center))
                    if Rred_generators is not None:
                        Rred_radius = np.sum(np.abs(Rred_generators), axis=1)
                        Rred_radius_max = np.max(Rred_radius) if Rred_radius.size > 0 else 0
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}: Rred center max = {Rred_center_max:.6e}, Rred radius max = {Rred_radius_max:.6e}", flush=True)
                    else:
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}: Rred center max = {Rred_center_max:.6e}", flush=True)
            except Exception as e:
                print(f"[priv_abstractionError_adaptive] Step {options['i']}: Error checking Rred: {e}", flush=True)
        
        Z = Rred.cartProd_(U)
        # Debug: Check Z size before quadMap
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
            try:
                Z_center = Z.center() if hasattr(Z, 'center') else None
                Z_generators = Z.generators() if hasattr(Z, 'generators') else None
                if Z_center is not None:
                    Z_center_max = np.max(np.abs(Z_center))
                    if Z_generators is not None:
                        Z_radius = np.sum(np.abs(Z_generators), axis=1)
                        Z_radius_max = np.max(Z_radius) if Z_radius.size > 0 else 0
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}: Z center max = {Z_center_max:.6e}, Z radius max = {Z_radius_max:.6e}", flush=True)
                    else:
                        print(f"[priv_abstractionError_adaptive] Step {options['i']}: Z center max = {Z_center_max:.6e}", flush=True)
                # Check H
                H_max = 0
                if H is not None:
                    if isinstance(H, (list, tuple)):
                        for h in H:
                            if h is not None and hasattr(h, 'size') and h.size > 0:
                                H_max = max(H_max, np.max(np.abs(h)))
                    elif hasattr(H, 'size') and H.size > 0:
                        H_max = np.max(np.abs(H))
                print(f"[priv_abstractionError_adaptive] Step {options['i']}: H_max = {H_max:.6e}", flush=True)
            except Exception as e:
                print(f"[priv_abstractionError_adaptive] Step {options['i']}: Error checking Z/H: {e}", flush=True)
        errorSec = 0.5 * Z.quadMap(H)
        # Check for infinite values in errorSec
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
            try:
                errorSec_center = errorSec.center() if hasattr(errorSec, 'center') else None
                if errorSec_center is not None and np.any(np.isinf(errorSec_center)):
                    print(f"[priv_abstractionError_adaptive] ERROR: errorSec center contains inf at step {options['i']}", flush=True)
                    raise CORAerror('CORA:reachSetExplosion', 'errorSec contains infinite values.')
            except Exception as e:
                if isinstance(e, CORAerror):
                    raise
                pass

        try:
            if options.get('thirdOrderTensorempty', False) or not callable(getattr(nlnsys, 'thirdOrderTensor', None)):
                ind = []
                options['thirdOrderTensorempty'] = True
            else:
                T, ind = nlnsys.thirdOrderTensor(totalInt_x, totalInt_u)
        except CORAerror as exc:
            if exc.identifier == 'interval:setoutofdomain':
                raise CORAerror('reach:setoutofdomain', 'Interval Arithmetic: Set out of domain.')
            raise

        if ind and np.any([len(i) > 0 for i in ind]):
            errorLagr = Interval(np.zeros((nlnsys.nr_of_dims, 1)), np.zeros((nlnsys.nr_of_dims, 1)))
            for i in range(len(ind)):
                error_sum = Interval(0, 0)
                for j in range(len(ind[i])):
                    error_tmp = dz.T @ T[i, ind[i][j]] @ dz
                    error_sum = error_sum + error_tmp * dz[ind[i][j]]
                errorLagr[i, 0] = (1 / 6) * error_sum
            errorLagr = Zonotope(errorLagr)
        else:
            errorLagr = 0
            options['thirdOrderTensorempty'] = True

        VerrorDyn = errorSec + errorLagr
        
        # Debug: Check components before reduction
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
            try:
                errorSec_center = errorSec.center() if hasattr(errorSec, 'center') else None
                errorLagr_center = errorLagr.center() if hasattr(errorLagr, 'center') else None
                VerrorDyn_center_before = VerrorDyn.center()
                print(f"[priv_abstractionError_adaptive] Step {options['i']}: errorSec center norm = {np.linalg.norm(errorSec_center) if errorSec_center is not None else 'N/A':.6e}, "
                      f"errorLagr center norm = {np.linalg.norm(errorLagr_center) if errorLagr_center is not None else 'N/A':.6e}, "
                      f"VerrorDyn center norm (before reduce) = {np.linalg.norm(VerrorDyn_center_before):.6e}", flush=True)
            except Exception:
                pass
        
        if 'gredIdx' in options and len(options['gredIdx'].get('VerrorDyn', [])) == options['i']:
            VerrorDyn = VerrorDyn.reduce('idx', options['gredIdx']['VerrorDyn'][options['i'] - 1])
        else:
            VerrorDyn_res = VerrorDyn.reduce('adaptive', 10 * options['redFactor'])
            if isinstance(VerrorDyn_res, tuple):
                VerrorDyn, _, idx = VerrorDyn_res
                if 'gredIdx' in options:
                    options['gredIdx'].setdefault('VerrorDyn', []).append(idx)
            else:
                VerrorDyn = VerrorDyn_res

        VerrorStat = []
        # Check if VerrorDyn contains infinite values
        VerrorDyn_center = VerrorDyn.center()
        if np.any(np.isinf(VerrorDyn_center)) or np.any(np.isnan(VerrorDyn_center)):
            # Debug: Print more info before raising error
            if options.get('progress', False):
                try:
                    R_center = R.center()
                    R_radius = np.linalg.norm(R.interval().rad())
                    print(f"[priv_abstractionError_adaptive] ERROR at step {options['i']}: VerrorDyn contains Inf/NaN. "
                          f"R center norm = {np.linalg.norm(R_center):.6e}, R radius = {R_radius:.6e}", flush=True)
                except Exception:
                    pass
            raise CORAerror('CORA:reachSetExplosion', 'VerrorDyn contains infinite or NaN values.')
        err = np.abs(VerrorDyn_center) + np.sum(np.abs(VerrorDyn.generators()), axis=1).reshape(-1, 1)

    # POLY --------------------------------------------------------------------
    elif options['alg'] == 'poly' and options['tensorOrder'] == 3:
        nlnsys = nlnsys.setHessian('standard')
        nlnsys = nlnsys.setThirdOrderTensor('int')

        dz = Interval.vertcat(IH_x, IH_u)

        Rred_diff_res = Zonotope(Rdiff).reduce('adaptive', np.sqrt(options['redFactor']))
        if isinstance(Rred_diff_res, tuple):
            Rred_diff = Rred_diff_res[0]
        else:
            Rred_diff = Rred_diff_res
        Z_diff = Rred_diff.cartProd_(U)

        error_secondOrder_dyn = Zdelta.quadMap(Z_diff, H) + 0.5 * Z_diff.quadMap(H)

        try:
            if options.get('thirdOrderTensorempty', False) or not callable(getattr(nlnsys, 'thirdOrderTensor', None)):
                ind = []
                options['thirdOrderTensorempty'] = True
            else:
                T, ind = nlnsys.thirdOrderTensor(totalInt_x, totalInt_u)
        except CORAerror as exc:
            if exc.identifier == 'interval:setoutofdomain':
                raise CORAerror('reach:setoutofdomain', 'Interval Arithmetic: Set out of domain.')
            raise

        if ind and np.any([len(i) > 0 for i in ind]):
            nrind = len(ind)
            error_thirdOrder_old = Interval(np.zeros((nrind, 1)), np.zeros((nrind, 1)))
            for i in range(nrind):
                for j in range(len(ind[i])):
                    error_thirdOrder_old[i, 0] = error_thirdOrder_old[i, 0] + \
                        (dz.T @ T[i, ind[i][j]] @ dz) * dz[ind[i][j]]
            error_thirdOrder_dyn = Zonotope((1 / 6) * error_thirdOrder_old)
        else:
            error_thirdOrder_dyn = 0
            options['thirdOrderTensorempty'] = True

        VerrorDyn = error_secondOrder_dyn + error_thirdOrder_dyn
        VerrorDyn_res = VerrorDyn.reduce('adaptive', np.sqrt(options['redFactor']))
        VerrorDyn = VerrorDyn_res[0] if isinstance(VerrorDyn_res, tuple) else VerrorDyn_res

        temp = VerrorDyn + Zonotope(VerrorStat)
        err = np.abs(temp.center()) + np.sum(np.abs(temp.generators()), axis=1).reshape(-1, 1)

    else:
        raise CORAerror('CORA:notSupported', 'Specified tensor order not supported.')

    return VerrorDyn, VerrorStat, err, options


def _aux_checkIfHessianConst(nlnsys: Any, options: Dict[str, Any], H: Any, totalInt_x: Any, totalInt_u: Any) -> Dict[str, Any]:
    # check if hessian is constant --- only once executed! (very first step)
    options['isHessianConst'] = True
    scalingFactor = 1.1
    H_test = nlnsys.hessian(totalInt_x.enlarge(scalingFactor), totalInt_u.enlarge(scalingFactor))
    for i in range(len(H)):
        if not np.all(H[i] == H_test[i]):
            options['isHessianConst'] = False
            break

    if options['isHessianConst']:
        options['hessianConst'] = []
        for i in range(len(H)):
            temp = abs(H[i])
            options['hessianConst'].append(np.maximum(temp.infimum(), temp.supremum()))

    options['hessianCheck'] = True
    return options
