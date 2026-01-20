"""
linReach_adaptive - computes the reachable set after linearization;
   based on automated tuning from [3] applied to algorithms [1,2]

Syntax:
    [Rti,Rtp,options] = linReach_adaptive(nlnsys,Rstart,params,options)

Inputs:
    nlnsys - nonlinearSys object
    Rstart - reachable set (time point of current time)
    params - model parameters
    options - struct with algorithm settings

Outputs:
    Rti - reachable set for time interval
    Rtp - reachable set for time point
    options - struct with algorithm settings

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       11-December-2020 (MATLAB)
Last update:   15-January-2021
               30-June-2021
               10-November-2023 (MW, improved estimate for finitehorizon)
Python translation: 2025
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contDynamics.nonlinearSys.private.priv_precompStatError_adaptive import priv_precompStatError_adaptive
from cora_python.contDynamics.nonlinearSys.private.priv_abstractionError_adaptive import priv_abstractionError_adaptive


def linReach_adaptive(nlnsys: Any, Rstart: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    # set linearization error
    error_adm = options['error_adm_horizon']
    # Debug: Check error_adm_horizon at start of step
    if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
        error_adm_max = np.max(error_adm) if error_adm.size > 0 else 0
        print(f"[linReach_adaptive] Step {options['i']}: Starting with error_adm_horizon max = {error_adm_max:.6e}", flush=True)

    lastStep = False
    veryfirststep = False
    timeStepequalHorizon = False

    if options['i'] == 1:
        veryfirststep = True
        options['timeStep'] = (params['tFinal'] - params['tStart']) * 0.01
        finitehorizon = options['timeStep']
        options['tt_err'] = []
        if options['alg'] == 'lin':
            options.setdefault('gredIdx', {})
            options['gredIdx'].setdefault('Rhomti', [])
            options['gredIdx'].setdefault('Rhomtp', [])
            options['gredIdx'].setdefault('Rpar', [])
            options['gredIdx'].setdefault('Rred', [])
            options['gredIdx'].setdefault('VerrorDyn', [])
    elif options['timeStep'] > params['tFinal'] - options['t']:
        lastStep = True
        options['timeStep'] = params['tFinal'] - options['t']
        finitehorizon = options['timeStep']
    elif options['i'] > 1:
        # MATLAB line 80-81: finitehorizon = options.finitehorizon(options.i-1) ...
        #     * (1 + options.varphi(options.i-1) - options.zetaphi(options.minorder+1));
        # MATLAB uses minorder+1 as index (1-indexed), Python uses minorder (0-indexed)
        # So if MATLAB uses options.zetaphi(options.minorder+1), Python uses options['zetaphi'][minorder]
        # Convert minorder to int for list indexing
        minorder = int(options.get('minorder', 0))
        finitehorizon = options['finitehorizon'][options['i'] - 2] * (
            1 + options['varphi'][options['i'] - 2] - options['zetaphi'][minorder])
        # MATLAB line 84: min([params.tFinal - options.t, finitehorizon]);
        # Note: MATLAB computes this but doesn't assign it (dead code or bug)
        # Python correctly assigns it to cap finitehorizon
        finitehorizon = min(params['tFinal'] - options['t'], finitehorizon)
        if withinTol(finitehorizon, options['stepsize'][options['i'] - 2], 1e-3):
            finitehorizon = options['finitehorizon'][options['i'] - 2] * 1.1
        options['timeStep'] = finitehorizon
        
        # MATLAB line 101: assert(options.timeStep > 0,'Tuning error.. report to devs');
        assert options['timeStep'] > 0, 'Tuning error.. report to devs'
        
        if options['i'] == 2 and options['alg'] == 'poly' and options.get('polyGain', False):
            options['orders'] = options['orders'] - 1
            options['minorder'] = np.min(options['orders'])

    zeroWidthDim = withinTol(np.sum(np.abs(Zonotope(Rstart).generators()), axis=1), 0)
    options['run'] = 0

    if veryfirststep and options['alg'] == 'lin':
        options = _aux_initStepTensorOrder(nlnsys, Rstart, params, options)
    options['run'] += 1

    abscount = 0

    while True:
        if options.get('progress', False) and options.get('run', 0) > 0:
            print(f"[linReach_adaptive] run={options.get('run', 0)} abscount={abscount} timeStep={options.get('timeStep', np.nan):.6g}", flush=True)
        nlnsys, linsys, linParams, linOptions = nlnsys.linearize(Rstart, params, options)

        if options['run'] == 1:
            zetaP = np.exp(np.trace(linsys.A * options['timeStep']))

        Rdelta = Rstart + (-nlnsys.linError.p.x)

        try:
            Rlin, options, linOptions = linsys.initReach_adaptive(Rdelta, options, linParams, linOptions)
        except Exception as exc:
            if getattr(exc, 'identifier', '') == 'expmtie:notconverging':
                options['timeStep'] = options['timeStep'] * 0.5
                finitehorizon = options['timeStep']
                continue
            raise

        if options['alg'] == 'poly' and options['run'] == 1:
            H, Zdelta, errorStat, T, ind3, Zdelta3 = priv_precompStatError_adaptive(nlnsys, Rdelta, params['U'], options)
        else:
            H = Zdelta = errorStat = T = ind3 = Zdelta3 = None

        Rlintp = Rlin['tp']
        Rlinti = Rlin['ti']

        perfIndCounter = 1
        perfInds = []
        options['Lconverged'] = False

        inner_iter = 0
        while True:
            inner_iter += 1
            if options.get('progress', False) and inner_iter % 100 == 0:
                print(f"[linReach_adaptive] inner loop iter={inner_iter} abscount={abscount} perfIndCounter={perfIndCounter}", flush=True)
            abscount += 1
            if np.any(error_adm):
                errG = np.diag(error_adm.flatten())
                Verror = Zonotope(np.zeros_like(error_adm), errG[:, np.any(errG, axis=0)])
                RallError, options = linsys.errorSolution_adaptive(options, Verror)
            else:
                RallError = Zonotope(np.zeros((nlnsys.nr_of_dims, 1)))

            try:
                if options['alg'] == 'lin':
                    # Debug: Check Rlinti and RallError before computing Rmax
                    if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] >= 340):
                        try:
                            Rlinti_center = Rlinti.center() if hasattr(Rlinti, 'center') else None
                            Rlinti_generators = Rlinti.generators() if hasattr(Rlinti, 'generators') else None
                            RallError_center = RallError.center() if hasattr(RallError, 'center') else None
                            RallError_generators = RallError.generators() if hasattr(RallError, 'generators') else None
                            if Rlinti_center is not None:
                                Rlinti_center_max = np.max(np.abs(Rlinti_center))
                                if Rlinti_generators is not None:
                                    Rlinti_radius = np.sum(np.abs(Rlinti_generators), axis=1)
                                    Rlinti_radius_max = np.max(Rlinti_radius) if Rlinti_radius.size > 0 else 0
                                    print(f"[linReach_adaptive] Step {options['i']}: Rlinti center max = {Rlinti_center_max:.6e}, Rlinti radius max = {Rlinti_radius_max:.6e}", flush=True)
                            if RallError_center is not None:
                                RallError_center_max = np.max(np.abs(RallError_center))
                                if RallError_generators is not None:
                                    RallError_radius = np.sum(np.abs(RallError_generators), axis=1)
                                    RallError_radius_max = np.max(RallError_radius) if RallError_radius.size > 0 else 0
                                    print(f"[linReach_adaptive] Step {options['i']}: RallError center max = {RallError_center_max:.6e}, RallError radius max = {RallError_radius_max:.6e}", flush=True)
                                    # Check if RallError is too large (set explosion)
                                    if RallError_radius_max > 1e+100:
                                        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                                        raise CORAerror('CORA:reachSetExplosion', f'RallError radius ({RallError_radius_max:.6e}) exceeds threshold. Set explosion detected.')
                        except Exception as e:
                            if isinstance(e, CORAerror) and e.identifier == 'CORA:reachSetExplosion':
                                raise
                            print(f"[linReach_adaptive] Step {options['i']}: Error checking Rlinti/RallError: {e}", flush=True)
                    Rmax = Rlinti + RallError
                    # Check if Rmax is too large before passing to priv_abstractionError_adaptive
                    try:
                        Rmax_center = Rmax.center() if hasattr(Rmax, 'center') else None
                        Rmax_generators = Rmax.generators() if hasattr(Rmax, 'generators') else None
                        if Rmax_center is not None and Rmax_generators is not None:
                            Rmax_radius = np.sum(np.abs(Rmax_generators), axis=1)
                            Rmax_radius_max = np.max(Rmax_radius) if Rmax_radius.size > 0 else 0
                            if Rmax_radius_max > 1e+100:
                                from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
                                raise CORAerror('CORA:reachSetExplosion', f'Rmax radius ({Rmax_radius_max:.6e}) exceeds threshold. Set explosion detected.')
                    except Exception as e:
                        if isinstance(e, CORAerror) and e.identifier == 'CORA:reachSetExplosion':
                            raise
                    Rdiff = Zonotope(np.zeros((len(RallError.center()), 1)))
                    Rtemp = Rdiff + RallError
                    VerrorDyn, VerrorStat, trueError, options = priv_abstractionError_adaptive(
                        nlnsys, Rmax, Rtemp, params['U'], options)
                else:
                    Rdiff = _aux_deltaReach_adaptive(linsys, Rdelta, linOptions)
                    try:
                        Rmax = Rdelta + Zonotope(Rdiff) + RallError
                    except Exception:
                        Rmax = Rdelta + Rdiff + RallError
                    VerrorDyn, VerrorStat, trueError, options = priv_abstractionError_adaptive(
                        nlnsys, Rmax, Rdiff + RallError, params['U'], options,
                        H, Zdelta, errorStat, T, ind3, Zdelta3)
            except Exception as exc:
                if getattr(exc, 'identifier', '') == 'reach:setoutofdomain':
                    options['Lconverged'] = False
                    break
                raise

            # MATLAB line 229: perfIndCurr = max(trueError ./ error_adm);
            # MATLAB uses element-wise division, which produces Inf for zero denominators
            # MATLAB does NOT handle Inf specially - it just uses it directly
            # If perfIndCurr is Inf, then perfIndCurr <= 1 is false, so loop continues
            with np.errstate(divide='ignore', invalid='ignore'):
                # Element-wise division: trueError ./ error_adm
                # This will produce Inf where error_adm is zero (if trueError is non-zero)
                perfIndCurr_ratio = trueError / error_adm
                # MATLAB: max() of the result - keep Inf if present, don't replace it
                perfIndCurr = np.max(perfIndCurr_ratio)
                # Only handle NaN case (when both error_adm and trueError are zero)
                if np.isnan(perfIndCurr):
                    # Both error_adm and trueError are zero - converged
                    perfIndCurr = 0
            
            if options.get('progress', False) and inner_iter % 100 == 0:
                # Debug: check if error_adm has zeros
                error_adm_has_zeros = np.any(error_adm == 0) if error_adm.size > 0 else False
                error_adm_shape = error_adm.shape if hasattr(error_adm, 'shape') else 'unknown'
                trueError_shape = trueError.shape if hasattr(trueError, 'shape') else 'unknown'
                print(f"[linReach_adaptive] perfIndCurr={perfIndCurr:.6g} (inf={np.isinf(perfIndCurr)}) perfIndCounter={perfIndCounter} error_adm_max={np.max(error_adm) if np.any(error_adm) else 0:.6g} trueError_max={np.max(trueError) if np.any(trueError) else 0:.6g} error_adm_has_zeros={error_adm_has_zeros} error_adm_shape={error_adm_shape} trueError_shape={trueError_shape}", flush=True)
            
            # MATLAB break conditions (lines 230-239):
            # 1. perfIndCurr <= 1 || ~any(trueError) -> converged
            # 2. perfIndCounter > 2 && perfInds(perfIndCounter) > perfInds(perfIndCounter-1) -> diverging
            if perfIndCurr <= 1 or not np.any(trueError):
                perfInds.append(perfIndCurr)
                options['Lconverged'] = True
                if options.get('progress', False):
                    print(f"[linReach_adaptive] Inner loop converged after {inner_iter} iterations, perfIndCurr={perfIndCurr:.6g}", flush=True)
                break
            elif perfIndCounter > 1:
                perfInds.append(perfIndCurr)
                if perfIndCounter > 2 and perfInds[-1] > perfInds[-2]:
                    options['Lconverged'] = False
                    if options.get('progress', False):
                        print(f"[linReach_adaptive] Inner loop diverging after {inner_iter} iterations, perfIndCurr={perfIndCurr:.6g}", flush=True)
                    break
            
            # MATLAB line 243: increase admissible abstraction error for next iteration
            # This updates error_adm to be 1.1 * trueError, which should eventually
            # make error_adm large enough that perfIndCurr <= 1
            error_adm = 1.1 * trueError
            perfIndCounter += 1

        if not options['Lconverged']:
            options['timeStep'] = options['timeStep'] * 0.5
            finitehorizon = options['timeStep']
            error_adm = options['error_adm_horizon']
            continue
        
        # ... now containment of L ensured
        
        # if rank of Rstart changes... re-compute orders
        if veryfirststep or not np.all(zeroWidthDim == options.get('zeroWidthDim', zeroWidthDim)):
            options['zeroWidthDim'] = zeroWidthDim
            options = _aux_getPowers(nlnsys, options, linsys, zeroWidthDim, VerrorDyn)
        
        # MATLAB line 264: compute set of abstraction errors
        # This is always computed, regardless of the condition above
        Rerror, options = linsys.errorSolution_adaptive(options, VerrorDyn, VerrorStat)

        # MATLAB line 267-279: measure abstraction error
        if isinstance(Rerror, PolyZonotope):
            # MATLAB: abstrerr = sum(abs(Rerror.GI),2)';
            # MATLAB produces a row vector (transpose), Python axis=1 produces column vector
            # We need to transpose to match MATLAB's row vector
            abstrerr = np.sum(np.abs(Rerror.GI), axis=1).reshape(1, -1)
        else:
            # MATLAB: abstrerr = sum(abs(generators(Rerror)),2)';
            abstrerr = np.sum(np.abs(Rerror.generators()), axis=1).reshape(1, -1)

        if options['run'] == 1 and not lastStep:
            if options['i'] == 1:
                if veryfirststep:
                    veryfirststep = False
                    abstrerr_h = abstrerr
                    Rerror_h = Rerror
                    Rti_h = Rlinti
                    Rtp_h = Rlintp
                    linx_h = nlnsys.linError.p.x
                    zetaP_h = zetaP
                    options['timeStep'] = options['decrFactor'] * options['timeStep']
                    error_adm = options['decrFactor'] * trueError
                    continue

                # MATLAB line 306-308: temp = abstrerr(options.orders == options.minorder) ./ ...
                #     abstrerr_h(options.orders == options.minorder);
                #     varphi = max( temp(~isnan(temp)) );
                # Note: abstrerr is a row vector in MATLAB (1, n), orders is a column vector (n, 1)
                # We need to flatten orders_mask to use it for indexing
                orders_mask = (options['orders'] == options['minorder']).flatten()
                if abstrerr.ndim == 2 and abstrerr.shape[0] == 1:
                    # Row vector (MATLAB style): abstrerr is (1, n), use [0, :] to get row
                    temp = abstrerr[0, orders_mask] / abstrerr_h[0, orders_mask]
                else:
                    # Column vector
                    temp = abstrerr[orders_mask] / abstrerr_h[orders_mask]
                temp = temp[~np.isnan(temp)]
                varphi = np.max(temp) if temp.size > 0 else 0.0
                
                # MATLAB line 311: if varphi < options.zetaphi(options.minorder+1)
                # MATLAB uses minorder+1 (1-indexed), Python uses minorder (0-indexed)
                # Convert minorder to int for list indexing
                minorder_idx = int(options['minorder'])
                if varphi < options['zetaphi'][minorder_idx]:
                    finitehorizon = options['timeStep']
                    options['timeStep'] = options['decrFactor'] * options['timeStep']
                    error_adm = options['decrFactor'] * trueError
                    abstrerr_h = abstrerr
                    Rerror_h = Rerror
                    Rti_h = Rlinti
                    Rtp_h = Rlintp
                    linx_h = nlnsys.linError.p.x
                    zetaP_h = zetaP
                    continue

                options.setdefault('varphi', [])
                options.setdefault('finitehorizon', [])
                if len(options['varphi']) < options['i']:
                    options['varphi'].append(varphi)
                else:
                    options['varphi'][options['i'] - 1] = varphi
                if len(options['finitehorizon']) < options['i']:
                    options['finitehorizon'].append(finitehorizon)
                else:
                    options['finitehorizon'][options['i'] - 1] = finitehorizon

                options['timeStep'], _ = _aux_optimaldeltat(
                    Rstart, Rerror_h, finitehorizon, varphi, zetaP_h, options
                )
                options['error_adm_horizon'] = trueError
                error_adm = np.zeros((nlnsys.nr_of_dims, 1))
            else:
                abstrerr_h = abstrerr
                Rerror_h = Rerror
                Rti_h = Rlinti
                Rtp_h = Rlintp
                linx_h = nlnsys.linError.p.x
                options['timeStep'], _ = _aux_optimaldeltat(
                    Rstart, Rerror_h, finitehorizon, options['varphi'][options['i'] - 2], zetaP, options
                )
                options['timeStep'] = min(options['timeStep'], params['tFinal'] - options['t'])
                options['error_adm_horizon'] = trueError
                error_adm = options['error_adm_Deltatopt']

        elif options['run'] == 2 or lastStep:
            # MATLAB line 356-364: run using tuned time step size
            if timeStepequalHorizon:
                # MATLAB line 357-359: temp = abstrerr(options.orders == options.minorder) ./ ...
                #     abstrerr_h(options.orders == options.minorder);
                #     options.varphi(options.i,1) = max( temp(~isnan(temp)) );
                # Flatten orders_mask to use for indexing
                orders_mask = (options['orders'] == options['minorder']).flatten()
                if abstrerr.ndim == 2 and abstrerr.shape[0] == 1:
                    # Row vector (MATLAB style): abstrerr is (1, n)
                    temp = abstrerr[0, orders_mask] / abstrerr_h[0, orders_mask]
                else:
                    # Column vector
                    temp = abstrerr[orders_mask] / abstrerr_h[orders_mask]
                # Ensure varphi array is long enough
                while len(options['varphi']) < options['i']:
                    options['varphi'].append(0.0)
                options['varphi'][options['i'] - 1] = np.max(temp[~np.isnan(temp)])
            elif not lastStep:
                # MATLAB line 361-363: options.varphi(options.i,1) = aux_varphiest(...)
                # Ensure varphi array is long enough
                while len(options['varphi']) < options['i']:
                    options['varphi'].append(0.0)
                options['varphi'][options['i'] - 1] = _aux_varphiest(
                    finitehorizon, options['timeStep'], Rerror_h, Rerror,
                    options['decrFactor'], options['orders'], options['minorder']
                )
            # MATLAB line 366-367: save finite horizon and linearization error for next step
            #     options.finitehorizon(options.i,1) = finitehorizon;
            #     options.error_adm_Deltatopt = trueError;
            # Ensure finitehorizon array is long enough
            while len(options['finitehorizon']) < options['i']:
                options['finitehorizon'].append(0.0)
            options['finitehorizon'][options['i'] - 1] = finitehorizon
            options['error_adm_Deltatopt'] = trueError

            if options['alg'] == 'lin':
                if options['i'] == 1:
                    options['kappa_deltat'] = options['timeStep']
                    options['kappa_abstrerr'] = np.linalg.norm(abstrerr)
                elif not lastStep:
                    options = _aux_nextStepTensorOrder(nlnsys, options,
                                                       np.linalg.norm(abstrerr), linsys,
                                                       Rmax, Rtemp, params['U'])
            break

        options['run'] += 1

        if options['timeStep'] == finitehorizon:
            timeStepequalHorizon = True
            options['timeStep'] = options['timeStep'] * options['decrFactor']

    if timeStepequalHorizon:
        options['timeStep'] = finitehorizon
        Rti = Rti_h + linx_h
        Rtp = Rtp_h + linx_h
        Rerror = Rerror_h
    else:
        Rti = Rlinti + nlnsys.linError.p.x
        Rtp = Rlintp + nlnsys.linError.p.x

    if isinstance(Rerror, PolyZonotope):
        Rti = Rti.exactPlus(Rerror)
        Rtp = Rtp.exactPlus(Rerror)
    else:
        Rti = Rti + Rerror
        Rtp = Rtp + Rerror
    
    # Debug: Track components of Rtp to identify which part is growing
    if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] <= 5):
        try:
            p_x_norm = np.linalg.norm(nlnsys.linError.p.x)
            Rlintp_center = Rlintp.center()
            Rlintp_norm = np.linalg.norm(Rlintp_center)
            Rerror_norm = np.linalg.norm(Rerror.center()) if hasattr(Rerror, 'center') else 0
            Rtp_center = Rtp.center()
            Rtp_norm = np.linalg.norm(Rtp_center)
            print(f"[linReach_adaptive] Step {options['i']}: p.x norm = {p_x_norm:.6e}, Rlintp center norm = {Rlintp_norm:.6e}, "
                  f"Rerror center norm = {Rerror_norm:.6e}, Rtp center norm = {Rtp_norm:.6e}", flush=True)
        except Exception as e:
            pass  # Silently skip if computation fails

    options.setdefault('timeStepequalHorizon', [])
    options.setdefault('abscount', [])
    options.setdefault('stepsize', [])
    if len(options['timeStepequalHorizon']) < options['i']:
        options['timeStepequalHorizon'].append(timeStepequalHorizon)
        options['abscount'].append(abscount)
        options['stepsize'].append(options['timeStep'])
    else:
        options['timeStepequalHorizon'][options['i'] - 1] = timeStepequalHorizon
        options['abscount'][options['i'] - 1] = abscount
        options['stepsize'][options['i'] - 1] = options['timeStep']

    return Rti, Rtp, options


def _aux_initStepTensorOrder(nlnsys: Any, Rstartset: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    if Rstartset.generators().size == 0:
        options['tensorOrder'] = 2
        return options

    # Ensure linError and linearization point container exist
    if nlnsys.linError is None:
        nlnsys.linError = type('obj', (object,), {})()
    if not hasattr(nlnsys.linError, 'p') or nlnsys.linError.p is None:
        nlnsys.linError.p = type('obj', (object,), {})()

    nlnsys.linError.p.u = params['U'].center()
    nlnsys.linError.p.x = Rstartset.center()
    Rdelta = Rstartset + (-nlnsys.linError.p.x)

    options['tensorOrder'] = 2
    _, _, L0_2, options = priv_abstractionError_adaptive(nlnsys, Rdelta, None, params['U'], options)
    options['tensorOrder'] = 3
    _, _, L0_3, options = priv_abstractionError_adaptive(nlnsys, Rdelta, None, params['U'], options)

    L0_2lin = L0_2[L0_2 != 0]
    L0_3lin = L0_3[L0_3 != 0]
    if np.all(L0_3lin / L0_2lin > options['zetaK']):
        options['tensorOrder'] = 2
    else:
        options['tensorOrder'] = 3
    return options


def _aux_nextStepTensorOrder(nlnsys: Any, options: Dict[str, Any], abstrerr: float, linsys: Any, Rmax: Any, Rtemp: Any, U: Any) -> Dict[str, Any]:
    timeStep = options['timeStep']
    timeStep_prev = options['kappa_deltat']
    varphi_lim = options['decrFactor'] ** (options['minorder'] + 1)
    varphitotal = _aux_varphitotal(options['varphi'][options['i'] - 1], varphi_lim,
                                   timeStep_prev, timeStep, options['decrFactor'])
    abstrerr_prev = options['kappa_abstrerr']
    abstrerr_est = varphitotal * abstrerr

    if options['tensorOrder'] == 2:
        if abs(abstrerr_est / abstrerr_prev - 1) > 1 - options['zetaK']:
            abstrerr_2 = abstrerr
            options['tensorOrder'] = 3
        else:
            return options
        VerrorDyn, _, err, _ = priv_abstractionError_adaptive(nlnsys, Rmax, Rtemp, U, options)
        Rerror, _ = linsys.errorSolution_adaptive(options, VerrorDyn)
        abstrerr_3 = np.linalg.norm(np.sum(np.abs(Rerror.generators()), axis=1))
        if not np.all(abstrerr_3 / abstrerr_2 > options['zetaK']):
            options['kappa_deltat'] = options['timeStep']
            options['kappa_abstrerr'] = abstrerr_3
            options['error_adm_Deltatopt'] = err
            options['error_adm_horizon'] = np.zeros((nlnsys.nr_of_dims, 1))
        else:
            options['tensorOrder'] = 2
    else:
        if abs(abstrerr_est / abstrerr_prev - 1) > 1 - options['zetaK']:
            abstrerr_3 = abstrerr
            options['tensorOrder'] = 2
        else:
            return options
        VerrorDyn, _, err, _ = priv_abstractionError_adaptive(nlnsys, Rmax, Rtemp, U, options)
        Rerror, _ = linsys.errorSolution_adaptive(options, VerrorDyn)
        abstrerr_2 = np.linalg.norm(np.sum(np.abs(Rerror.generators()), axis=1))
        if np.all(abstrerr_3 / abstrerr_2 > options['zetaK']):
            options['kappa_deltat'] = options['timeStep']
            options['kappa_abstrerr'] = abstrerr_2
            options['error_adm_Deltatopt'] = err
            options['error_adm_horizon'] = np.zeros((nlnsys.nr_of_dims, 1))
        else:
            options['tensorOrder'] = 3

    return options


def _aux_varphitotal(varphi: float, varphi_lim: float, timeStep_prev: float, timeStep_curr: float, decrFactor: float) -> float:
    nrScalings = np.log(timeStep_prev / timeStep_curr) / np.log(decrFactor)
    varphis = [varphi]
    if nrScalings > 0:
        nrScalings_low = int(np.floor(nrScalings + 10 * np.finfo(float).eps))
        nrScalings_high = nrScalings_low + 1
        for i in range(1, nrScalings_low + 1):
            varphis.append(varphi + (varphi_lim - varphi) *
                           (timeStep_curr - decrFactor**i * timeStep_curr) / (timeStep_curr - 0))
        varphis[-1] = varphis[-1] + (1 - varphis[-1]) * (
            (timeStep_prev - timeStep_curr * decrFactor**nrScalings_high) /
            (timeStep_curr * (decrFactor**nrScalings_low - decrFactor**nrScalings_high))
        )
    else:
        nrScalings_low = int(np.ceil(nrScalings - 10 * np.finfo(float).eps))
        nrScalings_high = nrScalings_low - 1
        for i in range(-1, nrScalings_low - 1, -1):
            varphis.append(varphi + (varphi_lim - varphi) *
                           (timeStep_curr - decrFactor**i * timeStep_curr) / (timeStep_curr - 0))
        varphis[-1] = varphis[-1] + (1 - varphis[-1]) * (
            (timeStep_prev - timeStep_curr * decrFactor**nrScalings_high) /
            (timeStep_curr * (decrFactor**nrScalings_low - decrFactor**nrScalings_high))
        )
        varphis = [1 / v for v in varphis]

    return float(np.prod(varphis))


def _aux_optimaldeltat(Rt: Any, Rerr: Any, deltat: float, varphimin: float, zetaP: float, opt: Dict[str, Any]) -> Tuple[float, float]:
    mu = opt['decrFactor']
    dHused = 0.5 if opt['alg'] == 'lin' else 0.3
    zetaZ = opt['redFactor'] * dHused
    kprimemax = int(np.ceil(-np.log(100) / np.log(mu)))
    kprime = np.arange(0, kprimemax + 1)
    k = mu ** (-kprime)
    deltats = deltat * mu ** kprime
    floork = np.floor(k)

    if isinstance(Rt, Zonotope) and isinstance(Rerr, Zonotope):
        rR = np.linalg.norm(np.sum(np.abs(Rt.generators()), axis=1), 2)
        rerr1 = np.linalg.norm(np.sum(np.abs(Rerr.generators()), axis=1), 2)
    else:
        rR = np.linalg.norm(Rt.interval().rad())
        rerr1 = np.linalg.norm(np.sum(np.abs(Rerr.GI), axis=1), 2)

    varphimax = mu
    varphi_h = (varphimax - varphimin)
    varphi = (varphimin + (deltats[0] - deltats) / deltats[0] * varphi_h) / mu
    varphiprod = np.cumprod(varphi)

    sumallbutlast = np.zeros(len(floork))
    for i in range(len(floork)):
        firstfactor = (1 + 2 * zetaZ) ** (k[i] + 1 - np.arange(1, int(floork[i]) + 1))
        secondfactor = zetaP ** (1 - np.arange(1, int(floork[i]) + 1) / k[i])
        sumallbutlast[i] = np.sum(firstfactor * secondfactor)

    objfuncset = rR * (1 + 2 * zetaZ) ** k * zetaP + rerr1 / k * varphiprod * (
        sumallbutlast + (1 + zetaZ) ** (k - kprime) * (k - floork)
    )
    bestIdxnew = int(np.argmin(objfuncset))
    deltatest = deltats[bestIdxnew]
    # MATLAB line 648: kprimeest = bestIdxnew - 1;
    # MATLAB: bestIdxnew is 1-indexed, so kprimeest = bestIdxnew - 1 gives the value
    # Python: bestIdxnew is 0-indexed, so kprimeest = kprime[bestIdxnew] gives the value
    kprimeest = kprime[bestIdxnew]
    return float(deltatest), float(kprimeest)


def _aux_varphiest(horizon: float, deltat: float, Rerr_h: Any, Rerr_deltat: Any,
                   decrFactor: float, orders: np.ndarray, minorder: float) -> float:
    if isinstance(Rerr_h, Zonotope) and isinstance(Rerr_deltat, Zonotope):
        G_Rerr_h = Rerr_h.generators()
        G_Rerr_deltat = Rerr_deltat.generators()
        # orders is a column vector (2D), need to flatten for boolean indexing
        orders_mask = (orders.flatten() == minorder)
        rerr1 = np.linalg.norm(np.sum(np.abs(G_Rerr_h[orders_mask, :]), axis=1), 2)
        rerrk = np.linalg.norm(np.sum(np.abs(G_Rerr_deltat[orders_mask, :]), axis=1), 2)
    else:
        rerr1 = np.linalg.norm(np.sum(np.abs(Rerr_h.GI), axis=1), 2)
        rerrk = np.linalg.norm(np.sum(np.abs(Rerr_deltat.GI), axis=1), 2)

    rhs = rerrk / rerr1
    varphi_lim = decrFactor ** (minorder + 1)

    # MATLAB line 677: assert(rerr1 > rerrk,'Check abstraction errors');
    assert rerr1 > rerrk, 'Check abstraction errors'

    varphi_up = decrFactor ** (minorder + 1)
    varphi_low = 0
    cnt = 0

    while True:
        cnt += 1
        varphi = varphi_low + 0.5 * (varphi_up - varphi_low)
        varphitotal = _aux_varphitotal(varphi, varphi_lim, deltat, horizon, decrFactor)
        residual = varphitotal - rhs
        if residual < 0:
            varphi_low = varphi
        else:
            varphi_up = varphi
        if cnt == 10000:
            raise Exception("Bug in varphi estimation... report to devs")
        if abs(residual) < 1e-9 or (cnt > 1 and abs(varphi - prev_varphi) < 1e-6):
            return varphi
        prev_varphi = varphi


def _aux_getPowers(nlnsys: Any, options: Dict[str, Any], linsys: Any, zeroWidthDim: np.ndarray, Vdyn: Any) -> Dict[str, Any]:
    if options['i'] == 1 and 'orders' in options:
        options['minorder'] = np.min(options['orders'])
        return options

    A = linsys.A
    n = nlnsys.nr_of_dims

    if np.all(zeroWidthDim):
        sigma = 2 * np.ones((n, 1))
    elif not np.any(zeroWidthDim):
        sigma = np.zeros((n, 1))
    else:
        sigma = np.zeros((n, 1))
        sigmafound = np.zeros((n, 1), dtype=bool)
        if options.get('isHessianConst', False):
            Hess = options['hessianConst']
        else:
            if options['alg'] == 'poly':
                Hess = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u)
            else:
                nlnsys = nlnsys.setHessian('standard')
                Hess = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u)

        for i in range(n):
            H = Hess[i]
            for j in range(n):
                if not zeroWidthDim[j] and H[j, j] != 0:
                    sigma[i] = 0
                    sigmafound[i] = True
                    break
            if sigmafound[i]:
                continue
            sigma[i] = np.inf
            for j in range(n):
                for jj in range(j + 1, n):
                    if H[j, jj] + H[jj, j] != 0:
                        sigma[i] = min(sigma[i], np.count_nonzero(zeroWidthDim[[j, jj]]))
                        sigmafound[i] = True
            if not sigmafound[i]:
                sigma[i] = 2

    if options['alg'] == 'poly':
        options['polyGain'] = False
        if options.get('thirdOrderTensorempty', False):
            sigma = sigma + 1
            options['polyGain'] = True

    Gzero = np.linalg.norm(Vdyn.generators(), ord=1, axis=1) == 0
    qi = np.full((n, 1), np.inf)
    p = 0
    Apower = np.eye(n)
    qip = []
    while True:
        qip_p = np.full((n, 1), np.inf)
        for i in range(n):
            for j in range(n):
                if not Gzero[j] and Apower[i, j] != 0:
                    qip_p[i] = min(qip_p[i], sigma[j])
        qi = np.minimum(qi, qip_p + p)
        if np.all(qi < p + 1):
            break
        p += 1
        Apower = Apower @ A
        if p == 100:
            if np.all(qi[~np.isinf(qi)] < p + 1):
                break
    # MATLAB line 892: options.orders = qi; (column vector)
    # Python: keep as column vector to match MATLAB
    options['orders'] = qi
    options['minorder'] = np.min(options['orders'])
    return options


def _aux_deltaReach_adaptive(linsys: Any, Rinit: Any, options: Dict[str, Any]) -> Any:
    eAt = linsys.taylor.eAt
    F = linsys.taylor.F
    inputCorr = linsys.taylor.inputCorr
    Rtrans = linsys.taylor.Rtrans

    dim = len(F)
    Rhom_tp_delta = (eAt - np.eye(dim)) * Rinit + Rtrans

    # MATLAB line 913: O = polyZonotope(zeros(dim,1),[],[],[]);
    # Empty PolyZonotope: c=zeros(dim,1), G=[], GI=[], E=[]
    # In Python: use np.array([]).reshape(dim, 0) for empty G and GI to match constructor pattern
    O = PolyZonotope(np.zeros((dim, 1)), np.array([]).reshape(dim, 0), np.array([]).reshape(dim, 0), np.zeros((0, 0), dtype=int))
    Rdelta = O.enclose(Rhom_tp_delta) + F * Zonotope(Rinit) + inputCorr
    if options.get('isRV', False):
        Rdelta = Rdelta + linsys.taylor.RV
    return Rdelta
