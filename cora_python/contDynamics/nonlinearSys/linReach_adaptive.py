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
        prev_finitehorizon = options['finitehorizon'][options['i'] - 2]
        prev_varphi = options['varphi'][options['i'] - 2]
        zetaphi_val = options['zetaphi'][minorder]
        finitehorizon = prev_finitehorizon * (1 + prev_varphi - zetaphi_val)
        
        # Track key values for debugging (can be removed later)
        if options.get('i', 0) <= 20 or (options.get('i', 0) % 50 == 0):
            remTime = params['tFinal'] - options['t']
            capped_finitehorizon = min(remTime, finitehorizon)
            options.setdefault('_debug_finitehorizon', []).append({
                'step': options['i'],
                'prev_finitehorizon': prev_finitehorizon,
                'prev_varphi': prev_varphi,
                'zetaphi': zetaphi_val,
                'computed_finitehorizon': finitehorizon,
                'remTime': remTime,
                'capped_finitehorizon': capped_finitehorizon,
            })
        
        # MATLAB line 84: min([params.tFinal - options.t, finitehorizon]);
        # Note: MATLAB computes this but doesn't assign it (dead code or bug)
        # Python matches MATLAB behavior exactly - do not assign result
        min(params['tFinal'] - options['t'], finitehorizon)
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

    # Intermediate value tracking for comparison with MATLAB
    # Open trace file once before the while loop to capture both run 1 and run 2
    trace_values = options.get('traceIntermediateValues', False)
    trace_file = None
    if trace_values:
        trace_file = open(f'intermediate_values_step{options["i"]}_inner_loop.txt', 'w')
        trace_file.write(f'=== Inner Loop Intermediate Values - Step {options["i"]} ===\n')
        trace_file.write(f'Initial error_adm_horizon: {options.get("error_adm_horizon", "N/A")}\n')
        trace_file.write(f'Algorithm: {options.get("alg", "N/A")}\n')
        trace_file.write(f'TensorOrder: {options.get("tensorOrder", "N/A")}\n\n')

    while True:
        # Write run number at start of each run
        if trace_file:
            trace_file.write(f'\n=== Run {options.get("run", "N/A")} ===\n')
            trace_file.flush()
        
        # Track Rstart (input to linReach_adaptive for this step) if requested
        if options.get('trackUpstream', False):
            try:
                Rstart_center = Rstart.center() if hasattr(Rstart, 'center') else None
                Rstart_gens = Rstart.generators() if hasattr(Rstart, 'generators') else None
                if Rstart_center is not None and Rstart_gens is not None:
                    options['Rstart_tracking'] = {
                        'center': np.asarray(Rstart_center).copy(),
                        'generators': np.asarray(Rstart_gens).copy(),
                        'num_generators': Rstart_gens.shape[1] if Rstart_gens.ndim > 1 else 0
                    }
            except Exception as e:
                if options.get('progress', False):
                    print(f"[linReach_adaptive] Error tracking Rstart: {e}", flush=True)
        
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
        
        # Track Rlintp if requested (before translation and Rerror addition)
        if options.get('trackUpstream', False):
            try:
                Rlintp_center = Rlintp.center() if hasattr(Rlintp, 'center') else None
                Rlintp_gens = Rlintp.generators() if hasattr(Rlintp, 'generators') else None
                if Rlintp_center is not None and Rlintp_gens is not None:
                    options['Rlintp_tracking'] = {
                        'center': np.asarray(Rlintp_center).copy(),
                        'generators': np.asarray(Rlintp_gens).copy(),
                        'num_generators': Rlintp_gens.shape[1] if Rlintp_gens.ndim > 1 else 0
                    }
            except Exception as e:
                if options.get('progress', False):
                    print(f"[linReach_adaptive] Error tracking Rlintp: {e}", flush=True)

        perfIndCounter = 1
        perfInds = []
        options['Lconverged'] = False

        inner_iter = 0
        
        while True:
            inner_iter += 1
            if options.get('progress', False) and inner_iter % 100 == 0:
                print(f"[linReach_adaptive] inner loop iter={inner_iter} abscount={abscount} perfIndCounter={perfIndCounter}", flush=True)
            abscount += 1
            
            # Track intermediate values at start of iteration
            if trace_file:
                trace_file.write(f'\n--- Inner Loop Iteration {inner_iter} ---\n')
                trace_file.write(f'error_adm: {error_adm.flatten()}\n')
                trace_file.write(f'error_adm_max: {np.max(error_adm) if error_adm.size > 0 else 0:.15e}\n')
            
            if np.any(error_adm):
                errG = np.diag(error_adm.flatten())
                Verror = Zonotope(np.zeros_like(error_adm), errG[:, np.any(errG, axis=0)])
                RallError, options = linsys.errorSolution_adaptive(options, Verror)
                
                # Track RallError
                if trace_file:
                    try:
                        RallError_center = RallError.center()
                        RallError_gens = RallError.generators()
                        RallError_radius = np.sum(np.abs(RallError_gens), axis=1)
                        RallError_radius_max = np.max(RallError_radius) if RallError_radius.size > 0 else 0
                        trace_file.write(f'RallError center: {RallError_center.flatten()}\n')
                        trace_file.write(f'RallError radius: {RallError_radius.flatten()}\n')
                        trace_file.write(f'RallError radius_max: {RallError_radius_max:.15e}\n')
                    except Exception as e:
                        trace_file.write(f'RallError tracking error: {e}\n')
            else:
                RallError = Zonotope(np.zeros((nlnsys.nr_of_dims, 1)))
                if trace_file:
                    trace_file.write('RallError: zero (error_adm is zero)\n')

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
                    
                    # Track Rmax, Rlinti, and RallError before calling priv_abstractionError_adaptive
                    if options.get('trackUpstream', False):
                        try:
                            Rmax_center = Rmax.center() if hasattr(Rmax, 'center') else None
                            Rmax_gens = Rmax.generators() if hasattr(Rmax, 'generators') else None
                            Rlinti_center = Rlinti.center() if hasattr(Rlinti, 'center') else None
                            Rlinti_gens = Rlinti.generators() if hasattr(Rlinti, 'generators') else None
                            RallError_center = RallError.center() if hasattr(RallError, 'center') else None
                            RallError_gens = RallError.generators() if hasattr(RallError, 'generators') else None
                            
                            if Rmax_center is not None and Rmax_gens is not None:
                                options['Rmax_before_reduction'] = {
                                    'center': np.asarray(Rmax_center).copy(),
                                    'generators': np.asarray(Rmax_gens).copy(),
                                    'num_generators': Rmax_gens.shape[1] if len(Rmax_gens.shape) > 1 else 0
                                }
                            if Rlinti_center is not None and Rlinti_gens is not None:
                                options['Rlinti_before_Rmax'] = {
                                    'center': np.asarray(Rlinti_center).copy(),
                                    'generators': np.asarray(Rlinti_gens).copy(),
                                    'num_generators': Rlinti_gens.shape[1] if len(Rlinti_gens.shape) > 1 else 0
                                }
                            if RallError_center is not None and RallError_gens is not None:
                                options['RallError_before_Rmax'] = {
                                    'center': np.asarray(RallError_center).copy(),
                                    'generators': np.asarray(RallError_gens).copy(),
                                    'num_generators': RallError_gens.shape[1] if len(RallError_gens.shape) > 1 else 0
                                }
                            
                            # Also track Rlinti and RallError separately for direct comparison
                            if Rlinti_center is not None and Rlinti_gens is not None:
                                options['Rlinti_tracking'] = {
                                    'center': np.asarray(Rlinti_center).copy(),
                                    'generators': np.asarray(Rlinti_gens).copy(),
                                    'num_generators': Rlinti_gens.shape[1] if len(Rlinti_gens.shape) > 1 else 0
                                }
                            
                            if RallError_center is not None and RallError_gens is not None:
                                options['RallError_tracking'] = {
                                    'center': np.asarray(RallError_center).copy(),
                                    'generators': np.asarray(RallError_gens).copy(),
                                    'num_generators': RallError_gens.shape[1] if len(RallError_gens.shape) > 1 else 0
                                }
                        except Exception as e:
                            if options.get('progress', False):
                                print(f"[linReach_adaptive] Step {options['i']}: Error tracking Rmax components: {e}", flush=True)
                    
                    # Track Rmax before calling priv_abstractionError_adaptive
                    if trace_file:
                        try:
                            Rmax_center = Rmax.center()
                            Rmax_gens = Rmax.generators()
                            Rmax_radius = np.sum(np.abs(Rmax_gens), axis=1)
                            Rmax_radius_max = np.max(Rmax_radius) if Rmax_radius.size > 0 else 0
                            trace_file.write(f'Rmax center: {Rmax_center.flatten()}\n')
                            trace_file.write(f'Rmax radius: {Rmax_radius.flatten()}\n')
                            trace_file.write(f'Rmax radius_max: {Rmax_radius_max:.15e}\n')
                        except Exception as e:
                            trace_file.write(f'Rmax tracking error: {e}\n')
                        # Flush file before calling priv_abstractionError_adaptive to ensure
                        # it can append Z/errorSec tracking data
                        trace_file.flush()
                    
                    Rdiff = Zonotope(np.zeros((len(RallError.center()), 1)))
                    Rtemp = Rdiff + RallError
                    VerrorDyn, VerrorStat, trueError, options = priv_abstractionError_adaptive(
                        nlnsys, Rmax, Rtemp, params['U'], options, trace_file=trace_file)
                    
                    # Track VerrorDyn and trueError after computation
                    if trace_file:
                        try:
                            VerrorDyn_center = VerrorDyn.center()
                            VerrorDyn_gens = VerrorDyn.generators()
                            VerrorDyn_radius = np.sum(np.abs(VerrorDyn_gens), axis=1)
                            VerrorDyn_radius_max = np.max(VerrorDyn_radius) if VerrorDyn_radius.size > 0 else 0
                            trace_file.write(f'VerrorDyn center: {VerrorDyn_center.flatten()}\n')
                            trace_file.write(f'VerrorDyn radius: {VerrorDyn_radius.flatten()}\n')
                            trace_file.write(f'VerrorDyn radius_max: {VerrorDyn_radius_max:.15e}\n')
                            trace_file.write(f'trueError: {trueError.flatten()}\n')
                            trace_file.write(f'trueError_max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                        except Exception as e:
                            trace_file.write(f'VerrorDyn/trueError tracking error: {e}\n')
                else:
                    Rdiff = _aux_deltaReach_adaptive(linsys, Rdelta, linOptions)
                    try:
                        Rmax = Rdelta + Zonotope(Rdiff) + RallError
                    except Exception:
                        Rmax = Rdelta + Rdiff + RallError
                    # Flush trace file before calling priv_abstractionError_adaptive
                    if trace_file:
                        trace_file.flush()
                    VerrorDyn, VerrorStat, trueError, options = priv_abstractionError_adaptive(
                        nlnsys, Rmax, Rdiff + RallError, params['U'], options,
                        H, Zdelta, errorStat, T, ind3, Zdelta3, trace_file=trace_file)
            except Exception as exc:
                if getattr(exc, 'identifier', '') == 'reach:setoutofdomain':
                    options['Lconverged'] = False
                    break
                raise

            # MATLAB line 229: perfIndCurr = max(trueError ./ error_adm);
            # MATLAB uses element-wise division, which produces Inf for zero denominators
            # MATLAB does NOT handle Inf/NaN specially - it just uses it directly
            # If perfIndCurr is Inf, then perfIndCurr <= 1 is false, so loop continues
            # If perfIndCurr is NaN, then perfIndCurr <= 1 is false, so loop continues
            # MATLAB's max() IGNORES NaN if other valid numbers exist, returns Inf if any element is Inf
            # NumPy's np.max() returns NaN if any element is NaN (different from MATLAB!)
            # Use np.nanmax() to match MATLAB's behavior (ignores NaN like MATLAB)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Element-wise division: trueError ./ error_adm
                # This will produce Inf where error_adm is zero (if trueError is non-zero)
                # This will produce NaN where both error_adm and trueError are zero
                perfIndCurr_ratio = trueError / error_adm
                # MATLAB: max() ignores NaN if other valid numbers exist
                # Use np.nanmax() to match MATLAB's behavior
                # If all values are NaN, np.nanmax() returns NaN (matches MATLAB when all NaN)
                # If any value is Inf, np.nanmax() returns Inf (matches MATLAB)
                perfIndCurr = np.nanmax(perfIndCurr_ratio)
                # Handle case where all values are NaN (both error_adm and trueError are zero everywhere)
                # In MATLAB: max([NaN, NaN]) = NaN, and NaN <= 1 is false, so loop continues
                # This is correct behavior - if both are zero, we should continue to next iteration
                # No special handling needed - let NaN propagate naturally
                
                # Track perfIndCurr computation
                if trace_file:
                    trace_file.write(f'perfIndCurr_ratio: {perfIndCurr_ratio.flatten()}\n')
                    trace_file.write(f'perfIndCurr: {perfIndCurr:.15e}\n')
                    trace_file.write(f'perfIndCurr isinf: {np.isinf(perfIndCurr)}\n')
                    trace_file.write(f'perfIndCurr isnan: {np.isnan(perfIndCurr)}\n')
                    trace_file.write(f'perfIndCurr <= 1: {perfIndCurr <= 1}\n')
            
            if options.get('progress', False) and inner_iter % 100 == 0:
                # Debug: check if error_adm has zeros
                error_adm_has_zeros = np.any(error_adm == 0) if error_adm.size > 0 else False
                error_adm_shape = error_adm.shape if hasattr(error_adm, 'shape') else 'unknown'
                trueError_shape = trueError.shape if hasattr(trueError, 'shape') else 'unknown'
                print(f"[linReach_adaptive] perfIndCurr={perfIndCurr:.6g} (inf={np.isinf(perfIndCurr)}) perfIndCounter={perfIndCounter} error_adm_max={np.max(error_adm) if np.any(error_adm) else 0:.6g} trueError_max={np.max(trueError) if np.any(trueError) else 0:.6g} error_adm_has_zeros={error_adm_has_zeros} error_adm_shape={error_adm_shape} trueError_shape={trueError_shape}", flush=True)
            
            # MATLAB break conditions (lines 230-239):
            # 1. perfIndCurr <= 1 || ~any(trueError) -> converged
            # 2. perfIndCounter > 2 && perfInds(perfIndCounter) > perfInds(perfIndCounter-1) -> diverging
            if trace_file:
                trace_file.write(f'perfIndCounter: {perfIndCounter}\n')
                trace_file.write(f'perfInds: {perfInds}\n')
                if perfIndCounter > 1:
                    trace_file.write(f'perfInds[-1] > perfInds[-2]: {perfInds[-1] > perfInds[-2] if len(perfInds) >= 2 else "N/A"}\n')
            
            if perfIndCurr <= 1 or not np.any(trueError):
                perfInds.append(perfIndCurr)
                options['Lconverged'] = True
                if trace_file:
                    trace_file.write('CONVERGED: perfIndCurr <= 1 or ~any(trueError)\n')
                if options.get('progress', False):
                    print(f"[linReach_adaptive] Inner loop converged after {inner_iter} iterations, perfIndCurr={perfIndCurr:.6g}", flush=True)
                break
            elif perfIndCounter > 1:
                perfInds.append(perfIndCurr)
                if perfIndCounter > 2 and perfInds[-1] > perfInds[-2]:
                    options['Lconverged'] = False
                    if trace_file:
                        trace_file.write('DIVERGING: perfIndCounter > 2 && perfInds increasing\n')
                    if options.get('progress', False):
                        print(f"[linReach_adaptive] Inner loop diverging after {inner_iter} iterations, perfIndCurr={perfIndCurr:.6g}", flush=True)
                    break
            
            # MATLAB line 243: increase admissible abstraction error for next iteration
            # This updates error_adm to be 1.1 * trueError, which should eventually
            # make error_adm large enough that perfIndCurr <= 1
            error_adm_old = error_adm.copy()
            error_adm = 1.1 * trueError
            perfIndCounter += 1
            
            if trace_file:
                trace_file.write(f'error_adm updated: {error_adm.flatten()}\n')
                trace_file.write(f'error_adm_max updated: {np.max(error_adm) if error_adm.size > 0 else 0:.15e}\n')

        if trace_file:
            trace_file.write(f'\n=== Inner Loop Complete ===\n')
            trace_file.write(f'Lconverged: {options["Lconverged"]}\n')
            trace_file.write(f'Total iterations: {inner_iter}\n')
            trace_file.write(f'Final perfIndCurr: {perfIndCurr if "perfIndCurr" in locals() else "N/A"}\n')
            # Don't close yet - we need it for run == 2 tracking
            trace_file.flush()
        
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
        
        # Track VerrorDyn before errorSolution_adaptive for comparison
        if options.get('trackUpstream', False):
            VerrorDyn_before_errorsolution = {
                'center': VerrorDyn.center().copy() if hasattr(VerrorDyn, 'center') else None,
                'generators': VerrorDyn.generators().copy() if hasattr(VerrorDyn, 'generators') else None,
            }
            if VerrorDyn_before_errorsolution['generators'] is not None:
                VerrorDyn_before_errorsolution['radius'] = np.sum(np.abs(VerrorDyn_before_errorsolution['generators']), axis=1)
                VerrorDyn_before_errorsolution['radius_max'] = np.max(VerrorDyn_before_errorsolution['radius']) if VerrorDyn_before_errorsolution['radius'].size > 0 else 0
            else:
                VerrorDyn_before_errorsolution['radius'] = None
                VerrorDyn_before_errorsolution['radius_max'] = None
        
        Rerror, options = linsys.errorSolution_adaptive(options, VerrorDyn, VerrorStat)
        
        # Track Rerror before it's used in _aux_optimaldeltat
        if options.get('trackUpstream', False):
            Rerror_before_optimaldeltat = {
                'center': Rerror.center().copy() if hasattr(Rerror, 'center') else None,
                'generators': Rerror.generators().copy() if hasattr(Rerror, 'generators') else None,
            }
            if Rerror_before_optimaldeltat['generators'] is not None:
                # MATLAB: radius = sum(abs(generators),2); rerr1 = vecnorm(radius,2). Use float64 for exact match.
                Rerror_before_optimaldeltat['radius'] = np.asarray(
                    np.sum(np.abs(Rerror_before_optimaldeltat['generators']), axis=1), dtype=np.float64
                )
                Rerror_before_optimaldeltat['radius_max'] = float(np.max(Rerror_before_optimaldeltat['radius'])) if Rerror_before_optimaldeltat['radius'].size > 0 else 0.0
                # Compute rerr1 equivalent (match MATLAB vecnorm(sum(abs(generators),2),2))
                Rerror_before_optimaldeltat['rerr1'] = float(np.linalg.norm(Rerror_before_optimaldeltat['radius'], 2))
            else:
                Rerror_before_optimaldeltat['radius'] = None
                Rerror_before_optimaldeltat['radius_max'] = None
                Rerror_before_optimaldeltat['rerr1'] = None
            
            # Store in options for later extraction (tt_err = Taylor order per step from errorSolution_adaptive)
            options.setdefault('upstreamLog', []).append({
                'step': options['i'],
                'run': options.get('run', 0),
                'VerrorDyn_before_errorsolution': VerrorDyn_before_errorsolution,
                'Rerror_before_optimaldeltat': Rerror_before_optimaldeltat,
                'tt_err': list(options.get('tt_err', [])),
            })

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
                    temp = np.asarray(abstrerr[0, orders_mask] / abstrerr_h[0, orders_mask], dtype=np.float64)
                else:
                    # Column vector
                    temp = np.asarray(abstrerr[orders_mask] / abstrerr_h[orders_mask], dtype=np.float64)
                temp = temp[~np.isnan(temp)]
                varphi = float(np.max(temp)) if temp.size > 0 else 0.0
                
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

                options['timeStep'], _, _optimaldeltat_bestIdx = _aux_optimaldeltat(
                    Rstart, Rerror_h, finitehorizon, varphi, zetaP_h, options
                )
                options['_optimaldeltat_bestIdx'] = _optimaldeltat_bestIdx
                options['error_adm_horizon'] = trueError
                
                # Track run == 1, i == 1: error_adm_horizon update
                if trace_file:
                    trace_file.write(f'\n=== Run 1, Step 1: error_adm_horizon Update ===\n')
                    trace_file.write(f'trueError: {trueError.flatten()}\n')
                    trace_file.write(f'trueError_max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                    trace_file.write(f'error_adm_horizon SET TO: {trueError.flatten()}\n')
                    trace_file.write(f'error_adm_horizon max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                    trace_file.flush()
                
                error_adm = np.zeros((nlnsys.nr_of_dims, 1))
            else:
                abstrerr_h = abstrerr
                Rerror_h = Rerror
                Rti_h = Rlinti
                Rtp_h = Rlintp
                linx_h = nlnsys.linError.p.x
                _timeStep_uncapped, _, _optimaldeltat_bestIdx = _aux_optimaldeltat(
                    Rstart, Rerror_h, finitehorizon, options['varphi'][options['i'] - 2], zetaP, options
                )
                options['timeStep'] = min(_timeStep_uncapped, params['tFinal'] - options['t'])
                options['_optimaldeltat_bestIdx'] = _optimaldeltat_bestIdx
                options['_timeStep_uncapped'] = _timeStep_uncapped
                options['error_adm_horizon'] = trueError
                
                # Track run == 1, i > 1: error_adm_horizon update
                if trace_file:
                    trace_file.write(f'\n=== Run 1, Step {options["i"]}: error_adm_horizon Update ===\n')
                    trace_file.write(f'trueError: {trueError.flatten()}\n')
                    trace_file.write(f'trueError_max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                    trace_file.write(f'error_adm_horizon SET TO: {trueError.flatten()}\n')
                    trace_file.write(f'error_adm_horizon max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                    trace_file.write(f'error_adm_Deltatopt (used for error_adm): {options.get("error_adm_Deltatopt", "N/A")}\n')
                    trace_file.flush()
                
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
                    temp = np.asarray(abstrerr[0, orders_mask] / abstrerr_h[0, orders_mask], dtype=np.float64)
                else:
                    # Column vector
                    temp = np.asarray(abstrerr[orders_mask] / abstrerr_h[orders_mask], dtype=np.float64)
                # Ensure varphi array is long enough; match MATLAB max(temp(~isnan(temp)))
                while len(options['varphi']) < options['i']:
                    options['varphi'].append(0.0)
                valid = temp[~np.isnan(temp)]
                options['varphi'][options['i'] - 1] = float(np.max(valid)) if valid.size > 0 else 0.0
            elif not lastStep:
                # MATLAB line 361-363: options.varphi(options.i,1) = aux_varphiest(...)
                # Ensure varphi array is long enough; result as float for exact match with MATLAB
                while len(options['varphi']) < options['i']:
                    options['varphi'].append(0.0)
                options['varphi'][options['i'] - 1] = float(_aux_varphiest(
                    finitehorizon, options['timeStep'], Rerror_h, Rerror,
                    options['decrFactor'], options['orders'], options['minorder']
                ))
            # MATLAB line 366-367: save finite horizon and linearization error for next step
            #     options.finitehorizon(options.i,1) = finitehorizon;
            #     options.error_adm_Deltatopt = trueError;
            # Ensure finitehorizon array is long enough
            while len(options['finitehorizon']) < options['i']:
                options['finitehorizon'].append(0.0)
            options['finitehorizon'][options['i'] - 1] = finitehorizon
            options['error_adm_Deltatopt'] = trueError
            
            # Track run == 2: error_adm_Deltatopt update
            if trace_file:
                trace_file.write(f'\n=== Run 2, Step {options["i"]}: error_adm_Deltatopt Update ===\n')
                trace_file.write(f'trueError: {trueError.flatten()}\n')
                trace_file.write(f'trueError_max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                trace_file.write(f'error_adm_Deltatopt SET TO: {trueError.flatten()}\n')
                trace_file.write(f'error_adm_Deltatopt max: {np.max(trueError) if trueError.size > 0 else 0:.15e}\n')
                trace_file.write(f'Current error_adm_horizon: {options.get("error_adm_horizon", "N/A")}\n')
                if isinstance(options.get("error_adm_horizon", None), np.ndarray):
                    trace_file.write(f'error_adm_horizon max: {np.max(options["error_adm_horizon"]) if options["error_adm_horizon"].size > 0 else 0:.15e}\n')
                trace_file.flush()

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

        # Check if we can reuse Run 1 results (optimizer chose full horizon)
        # MATLAB: if options.timeStep == finitehorizon
        # Use optimizer's chosen index (bestIdxnew == 0 => full horizon) so we are not sensitive to
        # floating point or capping: if the optimizer chose the first option, reuse Run 1 results.
        # Also accept exact equality of (uncapped) timeStep and finitehorizon for compatibility.
        bestIdx = options.get('_optimaldeltat_bestIdx', -1)
        uncapped_equals_horizon = options.get('_timeStep_uncapped', options['timeStep']) == finitehorizon
        if bestIdx == 0 or uncapped_equals_horizon:
            timeStepequalHorizon = True
            options['timeStep'] = options['timeStep'] * options['decrFactor']

    if timeStepequalHorizon:
        options['timeStep'] = finitehorizon
        Rti = Rti_h + linx_h
        Rtp = Rtp_h + linx_h
        Rerror = Rerror_h
        
        # Track that we used the timeStepequalHorizon path
        if options.get('trackUpstream', False):
            options['timeStepequalHorizon_used'] = True
            # Track Rtp_h and Rerror_h for debugging
            # These variables are set in earlier runs and should be available here
            try:
                # Rtp_h and Rerror_h are local variables, try to access them directly
                try:
                    Rtp_h_center = Rtp_h.center() if hasattr(Rtp_h, 'center') else None
                    Rtp_h_gens = Rtp_h.generators() if hasattr(Rtp_h, 'generators') else None
                    if Rtp_h_center is not None and Rtp_h_gens is not None:
                        options['Rtp_h_tracking'] = {
                            'center': np.asarray(Rtp_h_center).copy(),
                            'generators': np.asarray(Rtp_h_gens).copy(),
                            'num_generators': Rtp_h_gens.shape[1] if Rtp_h_gens.ndim > 1 else 0
                        }
                except NameError:
                    # Rtp_h not defined yet (shouldn't happen in timeStepequalHorizon path)
                    if options.get('progress', False):
                        print(f"[linReach_adaptive] Warning: Rtp_h not available for tracking", flush=True)
                
                try:
                    Rerror_h_center = Rerror_h.center() if hasattr(Rerror_h, 'center') else None
                    Rerror_h_gens = Rerror_h.generators() if hasattr(Rerror_h, 'generators') else None
                    if Rerror_h_center is not None and Rerror_h_gens is not None:
                        options['Rerror_h_tracking'] = {
                            'center': np.asarray(Rerror_h_center).copy(),
                            'generators': np.asarray(Rerror_h_gens).copy(),
                            'num_generators': Rerror_h_gens.shape[1] if Rerror_h_gens.ndim > 1 else 0
                        }
                except NameError:
                    # Rerror_h not defined yet (shouldn't happen in timeStepequalHorizon path)
                    if options.get('progress', False):
                        print(f"[linReach_adaptive] Warning: Rerror_h not available for tracking", flush=True)
            except Exception as e:
                if options.get('progress', False):
                    print(f"[linReach_adaptive] Error tracking Rtp_h/Rerror_h: {e}", flush=True)
            
            # Update the log entry directly since priv_abstractionError_adaptive
            # was already called in Run 1 and won't be called again
            upstream_log = options.get('upstreamLog', [])
            if upstream_log:
                # Find the last entry for this step
                for entry in reversed(upstream_log):
                    if entry.get('step') == options['i']:
                        if 'Rtp_h_tracking' in options:
                            entry['Rtp_h_tracking'] = options['Rtp_h_tracking']
                        if 'Rerror_h_tracking' in options:
                            entry['Rerror_h_tracking'] = options['Rerror_h_tracking']
                        # Also save initReach_tracking if available (from initReach_adaptive)
                        if 'initReach_tracking' in options:
                            entry['initReach_tracking'] = options['initReach_tracking']
                        # Also save Rlintp_tracking if available
                        if 'Rlintp_tracking' in options:
                            entry['Rlintp_tracking'] = options['Rlintp_tracking']
                        break
    else:
        Rti = Rlinti + nlnsys.linError.p.x
        Rtp = Rlintp + nlnsys.linError.p.x
        
        # Track that we used the normal path
        if options.get('trackUpstream', False):
            options['timeStepequalHorizon_used'] = False
            # Track Rerror (final value used for Rtp computation)
            try:
                Rerror_center = Rerror.center() if hasattr(Rerror, 'center') else None
                Rerror_gens = Rerror.generators() if hasattr(Rerror, 'generators') else None
                if Rerror_center is not None and Rerror_gens is not None:
                    options['Rerror_tracking'] = {
                        'center': np.asarray(Rerror_center).copy(),
                        'generators': np.asarray(Rerror_gens).copy(),
                        'num_generators': Rerror_gens.shape[1] if Rerror_gens.ndim > 1 else 0
                    }
            except Exception as e:
                if options.get('progress', False):
                    print(f"[linReach_adaptive] Error tracking Rerror: {e}", flush=True)
            
            # Update the log entry directly since priv_abstractionError_adaptive
            # was already called and won't pick up Rerror_tracking
            upstream_log = options.get('upstreamLog', [])
            if upstream_log:
                # Find the last entry for this step
                for entry in reversed(upstream_log):
                    if entry.get('step') == options['i']:
                        if 'Rerror_tracking' in options:
                            entry['Rerror_tracking'] = options['Rerror_tracking']
                        if 'Rlintp_tracking' in options:
                            entry['Rlintp_tracking'] = options['Rlintp_tracking']
                        # Also save initReach_tracking if available (from initReach_adaptive)
                        if 'initReach_tracking' in options:
                            entry['initReach_tracking'] = options['initReach_tracking']
                        break
            

    if isinstance(Rerror, PolyZonotope):
        Rti = Rti.exactPlus(Rerror)
        Rtp = Rtp.exactPlus(Rerror)
    else:
        Rti = Rti + Rerror
        Rtp = Rtp + Rerror
    
    # Track final Rtp (before reduction in reach_adaptive) if requested
    if options.get('trackUpstream', False):
        try:
            Rtp_center = Rtp.center() if hasattr(Rtp, 'center') else None
            Rtp_gens = Rtp.generators() if hasattr(Rtp, 'generators') else None
            if Rtp_center is not None and Rtp_gens is not None:
                options['Rtp_final_tracking'] = {
                    'center': np.asarray(Rtp_center).copy(),
                    'generators': np.asarray(Rtp_gens).copy(),
                    'num_generators': Rtp_gens.shape[1] if Rtp_gens.ndim > 1 else 0,
                    'timeStepequalHorizon_used': options.get('timeStepequalHorizon_used', False)
                }
        except Exception as e:
            if options.get('progress', False):
                print(f"[linReach_adaptive] Error tracking final Rtp: {e}", flush=True)
    
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
    
    # Close trace file now that all runs are complete
    if trace_file:
        try:
            trace_file.write(f'\n=== Step Complete ===\n')
            trace_file.write(f'Final error_adm_horizon: {options.get("error_adm_horizon", "N/A")}\n')
            if 'error_adm_Deltatopt' in options:
                trace_file.write(f'Final error_adm_Deltatopt: {options["error_adm_Deltatopt"]}\n')
            trace_file.close()
        except Exception:
            pass  # File may already be closed

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
    _, _, L0_2, options = priv_abstractionError_adaptive(nlnsys, Rdelta, None, params['U'], options, trace_file=None)
    options['tensorOrder'] = 3
    _, _, L0_3, options = priv_abstractionError_adaptive(nlnsys, Rdelta, None, params['U'], options, trace_file=None)

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
        VerrorDyn, _, err, _ = priv_abstractionError_adaptive(nlnsys, Rmax, Rtemp, U, options, trace_file=None)
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
        VerrorDyn, _, err, _ = priv_abstractionError_adaptive(nlnsys, Rmax, Rtemp, U, options, trace_file=None)
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


def _aux_optimaldeltat(Rt: Any, Rerr: Any, deltat: float, varphimin: float, zetaP: float, opt: Dict[str, Any]) -> Tuple[float, float, int]:
    mu = float(opt['decrFactor'])
    dHused = 0.5 if opt['alg'] == 'lin' else 0.3
    zetaZ = float(opt['redFactor'] * dHused)
    kprimemax = int(np.ceil(-np.log(100) / np.log(mu)))
    kprime = np.arange(0, kprimemax + 1)
    # MATLAB: k = mu.^(-kprime); deltats = deltat * mu.^kprime. Use float64 for exact match.
    k = np.asarray(mu ** (-kprime), dtype=np.float64)
    deltats = np.asarray(float(deltat) * (mu ** kprime), dtype=np.float64)
    floork = np.floor(k)

    if isinstance(Rt, Zonotope) and isinstance(Rerr, Zonotope):
        # MATLAB: rR = vecnorm(sum(abs(generators(Rt)),2),2);
        # MATLAB: generators(Rt) is (n, p) where n=dim, p=generators
        # MATLAB: sum(...,2) sums along dimension 2 (columns), producing (n, 1)
        # MATLAB: vecnorm(...,2) computes 2-norm along dimension 2, which is the 2-norm of that (n, 1) vector
        # Python: generators() returns (n, p), axis=1 sums along columns (generators), producing (n,)
        # Python: np.linalg.norm computes 2-norm of that (n,) array
        # Use float64 for consistency with MATLAB vecnorm(sum(abs(generators),2),2)
        rR = float(np.linalg.norm(np.asarray(np.sum(np.abs(Rt.generators()), axis=1), dtype=np.float64), 2))
        rerr1 = float(np.linalg.norm(np.asarray(np.sum(np.abs(Rerr.generators()), axis=1), dtype=np.float64), 2))
    else:
        rR = np.linalg.norm(Rt.interval().rad())
        rerr1 = np.linalg.norm(np.sum(np.abs(Rerr.GI), axis=1), 2)

    # MATLAB: varphimax = mu; varphi_h = (varphimax - varphimin); varphi = (...); varphiprod = cumprod(varphi)
    varphimax = mu
    varphi_h = float(varphimax - varphimin)
    varphi = np.asarray((float(varphimin) + (deltats[0] - deltats) / deltats[0] * varphi_h) / mu, dtype=np.float64)
    varphiprod = np.asarray(np.cumprod(varphi), dtype=np.float64)

    sumallbutlast = np.zeros(len(floork), dtype=np.float64)
    for i in range(len(floork)):
        firstfactor = (1 + 2 * zetaZ) ** (k[i] + 1 - np.arange(1, int(floork[i]) + 1))
        secondfactor = zetaP ** (1 - np.arange(1, int(floork[i]) + 1) / k[i])
        sumallbutlast[i] = float(np.sum(firstfactor * secondfactor))

    # MATLAB: objfuncset = rR * (1+2*zetaZ).^k * zetaP + rerr1./k .* varphiprod .* (...)
    objfuncset = np.asarray(
        rR * (1 + 2 * zetaZ) ** k * zetaP + rerr1 / k * varphiprod * (
            sumallbutlast + (1 + zetaZ) ** (k - kprime) * (k - floork)
        ),
        dtype=np.float64
    )
    bestIdxnew = int(np.argmin(objfuncset))
    deltatest = float(deltats[bestIdxnew])
    # MATLAB line 648: kprimeest = bestIdxnew - 1;
    # MATLAB: bestIdxnew is 1-indexed, so kprimeest = bestIdxnew - 1 gives the value
    # Python: bestIdxnew is 0-indexed, so kprimeest = kprime[bestIdxnew] gives the value
    kprimeest = kprime[bestIdxnew]
    
    # Track inputs and outputs for comparison with MATLAB
    if opt.get('trackOptimaldeltat', False):
        step = opt.get('i', 0)
        opt.setdefault('optimaldeltatLog', []).append({
            'step': step,
            'deltat': deltat,
            'varphimin': varphimin,
            'zetaP': zetaP,
            'rR': rR,
            'rerr1': rerr1,
            'varphiprod': varphiprod.tolist() if hasattr(varphiprod, 'tolist') else varphiprod,
            'deltats': deltats.tolist() if hasattr(deltats, 'tolist') else deltats,
            'objfuncset': objfuncset.tolist() if hasattr(objfuncset, 'tolist') else objfuncset,
            'bestIdxnew': bestIdxnew,
            'deltatest': deltatest,
            'kprimeest': kprimeest,
        })
    
    return float(deltatest), float(kprimeest), bestIdxnew


def _aux_varphiest(horizon: float, deltat: float, Rerr_h: Any, Rerr_deltat: Any,
                   decrFactor: float, orders: np.ndarray, minorder: float) -> float:
    if isinstance(Rerr_h, Zonotope) and isinstance(Rerr_deltat, Zonotope):
        G_Rerr_h = Rerr_h.generators()
        G_Rerr_deltat = Rerr_deltat.generators()
        # orders is a column vector (2D), need to flatten for boolean indexing
        orders_mask = (orders.flatten() == minorder)
        # MATLAB: rerr1 = vecnorm(sum(abs(G_Rerr_h(orders == minorder,:)),2),2); use float64 for exact match
        rerr1 = float(np.linalg.norm(np.asarray(np.sum(np.abs(G_Rerr_h[orders_mask, :]), axis=1), dtype=np.float64), 2))
        rerrk = float(np.linalg.norm(np.asarray(np.sum(np.abs(G_Rerr_deltat[orders_mask, :]), axis=1), dtype=np.float64), 2))
    else:
        rerr1 = float(np.linalg.norm(np.asarray(np.sum(np.abs(Rerr_h.GI), axis=1), dtype=np.float64), 2))
        rerrk = float(np.linalg.norm(np.asarray(np.sum(np.abs(Rerr_deltat.GI), axis=1), dtype=np.float64), 2))

    rhs = float(rerrk / rerr1)
    varphi_lim = float(decrFactor ** (minorder + 1))

    # MATLAB line 677: assert(rerr1 > rerrk,'Check abstraction errors');
    assert rerr1 > rerrk, 'Check abstraction errors'

    varphi_up = float(decrFactor ** (minorder + 1))
    varphi_low = 0.0
    cnt = 0

    while True:
        cnt += 1
        varphi = varphi_low + 0.5 * (varphi_up - varphi_low)
        varphitotal = float(_aux_varphitotal(varphi, varphi_lim, deltat, horizon, decrFactor))
        residual = float(varphitotal - rhs)
        if residual < 0:
            varphi_low = varphi
        else:
            varphi_up = varphi
        if cnt == 10000:
            raise Exception("Bug in varphi estimation... report to devs")
        if abs(residual) < 1e-9 or (cnt > 1 and abs(varphi - prev_varphi) < 1e-6):
            return float(varphi)
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
