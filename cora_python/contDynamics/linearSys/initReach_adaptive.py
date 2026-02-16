"""
initReach_adaptive - computes the linearized reachable set
   from an originally nonlinear system (linearization error = 0),
   loops until time step found which satisfies abstraction error bound

Syntax:
    Rend = initReach_adaptive(linsys,Rstart,options,linParams,linOptions)

Inputs:
    linsys - linearized system
    Rstart - reachable set of current time point
    options - options for nonlinear system
    linParams - model parameter for linearized system
    linOptions - options for linearized system

Outputs:
    Rend - reachable set (time interval/point) of current time + time step

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       25-May-2020 (MATLAB)
Last update:   --- (MATLAB)
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


def initReach_adaptive(linsys: Any, Rstart: Any, options: Dict[str, Any],
                       linParams: Dict[str, Any], linOptions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # ensure taylor container exists
    if getattr(linsys, 'taylor', None) is None:
        linsys.taylor = type('obj', (object,), {})()

    # reuse same truncation order as in last iteration of same step
    if 'tt_lin' in options and len(options['tt_lin']) == options['i']:
        linOptions['taylorTerms'] = options['tt_lin'][options['i'] - 1]

    # exponential matrix and time interval error (incl. adaptive taylorTerms)
    linsys, linOptions = _aux_expmtie_adaptive(linsys, linOptions)
    # compute reachable set due to input
    linsys, linOptions = _aux_inputSolution(linsys, linParams, linOptions)
    # change the time step
    linsys.taylor.timeStep = linOptions['timeStep']
    # save taylorTerms for next iteration of same step
    options.setdefault('tt_lin', [])
    if len(options['tt_lin']) < options['i']:
        options['tt_lin'].append(linOptions['taylorTerms'])
    else:
        options['tt_lin'][options['i'] - 1] = linOptions['taylorTerms']
    options.setdefault('etalinFro', [])
    if len(options['etalinFro']) < options['i']:
        options['etalinFro'].append(linOptions['etalinFro'])
    else:
        options['etalinFro'][options['i'] - 1] = linOptions['etalinFro']

    # compute reachable set of first time interval
    eAt = expm(linsys.A * linOptions['timeStep'])
    linsys.taylor.eAt = eAt

    F = linsys.taylor.F
    inputCorr = linsys.taylor.inputCorr
    Rtrans = linsys.taylor.Rtrans

    # first time step homogeneous solution
    Rhom_tp = eAt * Rstart + Rtrans
    Rhom = Rstart.enclose(Rhom_tp) + F * Rstart + inputCorr

    # Track initReach_adaptive computation if requested
    if options.get('trackUpstream', False):
        try:
            # Helper function to convert arrays to lists only if needed for pickle
            # For large arrays, keep as numpy arrays - pickle handles them efficiently
            def to_list_if_array(val):
                # Only convert small arrays or scalars to lists
                # Large arrays are kept as numpy arrays for performance
                if isinstance(val, np.ndarray):
                    if val.size < 100:  # Only convert small arrays
                        return val.tolist()
                    return val  # Keep large arrays as numpy arrays
                elif hasattr(val, 'tolist') and hasattr(val, 'size'):
                    if val.size < 100:
                        return val.tolist()
                    return val
                return val
            
            initReach_tracking = {
                'step': options.get('i'),
                'run': options.get('run', 0),
                'Rstart_center': to_list_if_array(np.asarray(Rstart.center())),
                'Rstart_generators': to_list_if_array(np.asarray(Rstart.generators())),
                'Rstart_num_generators': Rstart.generators().shape[1] if Rstart.generators().ndim > 1 else 0,
                'eAt': to_list_if_array(eAt),
                'F': to_list_if_array(F) if hasattr(F, 'tolist') else F,
                'inputCorr_center': to_list_if_array(np.asarray(inputCorr.center())),
                'inputCorr_generators': to_list_if_array(np.asarray(inputCorr.generators())),
                'inputCorr_num_generators': inputCorr.generators().shape[1] if inputCorr.generators().ndim > 1 else 0,
                'Rtrans_center': to_list_if_array(np.asarray(Rtrans.center())),
                'Rtrans_generators': to_list_if_array(np.asarray(Rtrans.generators())),
                'Rtrans_num_generators': Rtrans.generators().shape[1] if Rtrans.generators().ndim > 1 else 0,
                'Rhom_tp_center': to_list_if_array(np.asarray(Rhom_tp.center())),
                'Rhom_tp_generators': to_list_if_array(np.asarray(Rhom_tp.generators())),
                'Rhom_tp_num_generators': Rhom_tp.generators().shape[1] if Rhom_tp.generators().ndim > 1 else 0,
                'Rhom_center': to_list_if_array(np.asarray(Rhom.center())),
                'Rhom_generators': to_list_if_array(np.asarray(Rhom.generators())),
                'Rhom_num_generators': Rhom.generators().shape[1] if Rhom.generators().ndim > 1 else 0,
                'redFactor': options.get('redFactor'),
                'timeStep': linOptions.get('timeStep'),
            }
            options['initReach_tracking'] = initReach_tracking
            
            # REMOVED: File I/O on every call is extremely slow
            # Only write to file at the end if needed, or use a separate flag
            # The data is already in options['initReach_tracking'] and will be saved to upstreamLog
        except Exception as e:
            if options.get('progress', False):
                print(f"[initReach_adaptive] Error tracking: {e}", flush=True)

    Rend = {}

    # preliminary solutions without RV
    if 'gredIdx' in options and len(options['gredIdx'].get('Rhomti', [])) == options['i']:
        Rend['ti'] = Rhom.reduce('idx', options['gredIdx']['Rhomti'][options['i'] - 1])
    else:
        # Set debug attributes for reduction tracking
        if options.get('trackUpstream', False):
            Rhom._debug_step = options.get('i')
            Rhom._debug_run = options.get('run', 0)
            # Enable reduction details tracking
            Rhom._track_reduction_details = True
        Rhom_res = Rhom.reduce('adaptive', options['redFactor'])
        if isinstance(Rhom_res, tuple):
            Rend['ti'], _, idx = Rhom_res
            if 'gredIdx' in options:
                # Store at index options['i'] - 1 (0-based) to match MATLAB's {options.i} (1-based)
                gredIdx = options['gredIdx'].setdefault('Rhomti', [])
                # Ensure list is long enough
                while len(gredIdx) < options['i']:
                    gredIdx.append(None)
                gredIdx[options['i'] - 1] = idx
        else:
            Rend['ti'] = Rhom_res
        
        # Capture reduction details for Rend.ti if available
        if options.get('trackUpstream', False) and 'initReach_tracking' in options:
            if hasattr(Rend['ti'], '_reduction_details') and Rend['ti']._reduction_details is not None:
                reduction_details = Rend['ti']._reduction_details
                def to_list_if_array(val):
                    if isinstance(val, np.ndarray):
                        if val.size < 100:  # Only convert small arrays
                            return val.tolist()
                        return val  # Keep large arrays as numpy arrays
                    elif hasattr(val, 'tolist') and hasattr(val, 'size'):
                        if val.size < 100:
                            return val.tolist()
                        return val
                    return val
                options['initReach_tracking']['reduction_ti_diagpercent'] = reduction_details.get('diagpercent')
                options['initReach_tracking']['reduction_ti_dHmax'] = reduction_details.get('dHmax')
                options['initReach_tracking']['reduction_ti_nrG'] = reduction_details.get('nrG')
                options['initReach_tracking']['reduction_ti_last0Idx'] = reduction_details.get('last0Idx')
                options['initReach_tracking']['reduction_ti_h_computed'] = to_list_if_array(reduction_details.get('h_computed'))
                options['initReach_tracking']['reduction_ti_redIdx'] = reduction_details.get('redIdx')
                options['initReach_tracking']['reduction_ti_redIdx_0based'] = reduction_details.get('redIdx_0based')
                options['initReach_tracking']['reduction_ti_dHerror'] = reduction_details.get('dHerror')
                options['initReach_tracking']['reduction_ti_gredIdx'] = to_list_if_array(reduction_details.get('gredIdx'))
                options['initReach_tracking']['reduction_ti_gredIdx_len'] = reduction_details.get('gredIdx_len')
        
        # Clean up debug attributes
        if hasattr(Rhom, '_debug_step'):
            delattr(Rhom, '_debug_step')
        if hasattr(Rhom, '_debug_run'):
            delattr(Rhom, '_debug_run')
        if hasattr(Rhom, '_track_reduction_details'):
            delattr(Rhom, '_track_reduction_details')
    
    # Track Rend.ti after reduction if requested
    if options.get('trackUpstream', False) and 'initReach_tracking' in options:
        try:
            def to_list_if_array(val):
                if isinstance(val, np.ndarray):
                    if val.size < 100:  # Only convert small arrays
                        return val.tolist()
                    return val  # Keep large arrays as numpy arrays
                elif hasattr(val, 'tolist') and hasattr(val, 'size'):
                    if val.size < 100:
                        return val.tolist()
                    return val
                return val
            options['initReach_tracking']['Rend_ti_center'] = to_list_if_array(np.asarray(Rend['ti'].center()))
            options['initReach_tracking']['Rend_ti_generators'] = to_list_if_array(np.asarray(Rend['ti'].generators()))
            options['initReach_tracking']['Rend_ti_num_generators'] = Rend['ti'].generators().shape[1] if Rend['ti'].generators().ndim > 1 else 0
        except Exception as e:
            if options.get('progress', False):
                print(f"[initReach_adaptive] Error tracking Rend.ti: {e}", flush=True)

    if 'gredIdx' in options and len(options['gredIdx'].get('Rhomtp', [])) == options['i']:
        Rend['tp'] = Rhom_tp.reduce('idx', options['gredIdx']['Rhomtp'][options['i'] - 1])
    else:
        # Set debug attributes for reduction tracking
        if options.get('trackUpstream', False):
            Rhom_tp._debug_step = options.get('i')
            Rhom_tp._debug_run = options.get('run', 0)
            # Enable reduction details tracking
            Rhom_tp._track_reduction_details = True
        Rhomtp_res = Rhom_tp.reduce('adaptive', options['redFactor'])
        if isinstance(Rhomtp_res, tuple):
            Rend['tp'], _, idx = Rhomtp_res
            if 'gredIdx' in options:
                # Store at index options['i'] - 1 (0-based) to match MATLAB's {options.i} (1-based)
                gredIdx = options['gredIdx'].setdefault('Rhomtp', [])
                # Ensure list is long enough
                while len(gredIdx) < options['i']:
                    gredIdx.append(None)
                gredIdx[options['i'] - 1] = idx
        else:
            Rend['tp'] = Rhomtp_res
        
        # Capture reduction details if available
        if options.get('trackUpstream', False) and 'initReach_tracking' in options:
            reduction_details = None
            # Try to get from Z object first
            if hasattr(Rend['tp'], '_reduction_details') and Rend['tp']._reduction_details is not None:
                reduction_details = Rend['tp']._reduction_details
            # Fallback: read from file (more reliable)
            if reduction_details is None:
                try:
                    import pickle
                    import os
                    debug_file = 'reduceAdaptive_debug_python.pkl'
                    if os.path.exists(debug_file):
                        with open(debug_file, 'rb') as f:
                            try:
                                debug_data = pickle.load(f)
                                if debug_data and len(debug_data) > 0:
                                    # Get the last entry (most recent reduction)
                                    reduction_details = debug_data[-1]
                            except (EOFError, ValueError):
                                pass
                except Exception:
                    pass
            
            if reduction_details is not None:
                def to_list_if_array(val):
                    if isinstance(val, np.ndarray):
                        if val.size < 100:  # Only convert small arrays
                            return val.tolist()
                        return val  # Keep large arrays as numpy arrays
                    elif hasattr(val, 'tolist') and hasattr(val, 'size'):
                        if val.size < 100:
                            return val.tolist()
                        return val
                    return val
                options['initReach_tracking']['reduction_diagpercent'] = reduction_details.get('diagpercent')
                options['initReach_tracking']['reduction_dHmax'] = reduction_details.get('dHmax')
                options['initReach_tracking']['reduction_nrG'] = reduction_details.get('nrG')
                options['initReach_tracking']['reduction_last0Idx'] = reduction_details.get('last0Idx')
                options['initReach_tracking']['reduction_h_computed'] = to_list_if_array(reduction_details.get('h_computed'))
                options['initReach_tracking']['reduction_redIdx'] = reduction_details.get('redIdx')
                options['initReach_tracking']['reduction_redIdx_0based'] = reduction_details.get('redIdx_0based')
                options['initReach_tracking']['reduction_dHerror'] = reduction_details.get('dHerror')
                options['initReach_tracking']['reduction_gredIdx'] = to_list_if_array(reduction_details.get('gredIdx'))
                options['initReach_tracking']['reduction_gredIdx_len'] = reduction_details.get('gredIdx_len')
        
        # Clean up debug attributes
        if hasattr(Rhom_tp, '_debug_step'):
            delattr(Rhom_tp, '_debug_step')
        if hasattr(Rhom_tp, '_debug_run'):
            delattr(Rhom_tp, '_debug_run')
        if hasattr(Rhom_tp, '_track_reduction_details'):
            delattr(Rhom_tp, '_track_reduction_details')
    
    # Track Rend.tp after reduction if requested
    if options.get('trackUpstream', False) and 'initReach_tracking' in options:
        try:
            def to_list_if_array(val):
                if isinstance(val, np.ndarray):
                    if val.size < 100:  # Only convert small arrays
                        return val.tolist()
                    return val  # Keep large arrays as numpy arrays
                elif hasattr(val, 'tolist') and hasattr(val, 'size'):
                    if val.size < 100:
                        return val.tolist()
                    return val
                return val
            options['initReach_tracking']['Rend_tp_center'] = to_list_if_array(np.asarray(Rend['tp'].center()))
            options['initReach_tracking']['Rend_tp_generators'] = to_list_if_array(np.asarray(Rend['tp'].generators()))
            options['initReach_tracking']['Rend_tp_num_generators'] = Rend['tp'].generators().shape[1] if Rend['tp'].generators().ndim > 1 else 0
        except Exception as e:
            if options.get('progress', False):
                print(f"[initReach_adaptive] Error tracking Rend.tp: {e}", flush=True)

    # reduce and add RV only if exists
    if linOptions.get('isRV', False):
        if 'gredIdx' in options and len(options['gredIdx'].get('Rpar', [])) == options['i']:
            RV = linsys.taylor.RV.reduce('idx', options['gredIdx']['Rpar'][options['i'] - 1])
        else:
            RV_res = linsys.taylor.RV.reduce('adaptive', options['redFactor'])
            if isinstance(RV_res, tuple):
                RV, _, idx = RV_res
                if 'gredIdx' in options:
                    # Store at index options['i'] - 1 (0-based) to match MATLAB's {options.i} (1-based)
                    gredIdx = options['gredIdx'].setdefault('Rpar', [])
                    # Ensure list is long enough
                    while len(gredIdx) < options['i']:
                        gredIdx.append(None)
                    gredIdx[options['i'] - 1] = idx
            else:
                RV = RV_res

        Rend['ti'] = Rend['ti'] + RV
        Rend['tp'] = Rend['tp'] + RV
    
    # Track Rend.ti and Rend.tp after all modifications (reduction + RV if applicable)
    if options.get('trackUpstream', False) and 'initReach_tracking' in options:
        try:
            def to_list_if_array(val):
                if isinstance(val, np.ndarray):
                    if val.size < 100:  # Only convert small arrays
                        return val.tolist()
                    return val  # Keep large arrays as numpy arrays
                elif hasattr(val, 'tolist') and hasattr(val, 'size'):
                    if val.size < 100:
                        return val.tolist()
                    return val
                return val
            options['initReach_tracking']['Rend_ti_center'] = to_list_if_array(np.asarray(Rend['ti'].center()))
            options['initReach_tracking']['Rend_ti_generators'] = to_list_if_array(np.asarray(Rend['ti'].generators()))
            options['initReach_tracking']['Rend_ti_num_generators'] = Rend['ti'].generators().shape[1] if Rend['ti'].generators().ndim > 1 else 0
            options['initReach_tracking']['Rend_tp_center'] = to_list_if_array(np.asarray(Rend['tp'].center()))
            options['initReach_tracking']['Rend_tp_generators'] = to_list_if_array(np.asarray(Rend['tp'].generators()))
            options['initReach_tracking']['Rend_tp_num_generators'] = Rend['tp'].generators().shape[1] if Rend['tp'].generators().ndim > 1 else 0
            
            # REMOVED: File I/O on every call is extremely slow
            # The data is already in options['initReach_tracking'] and will be saved to upstreamLog
        except Exception as e:
            if options.get('progress', False):
                print(f"[initReach_adaptive] Error tracking Rend.ti/tp: {e}", flush=True)

    return Rend, options, linOptions


def _aux_expmtie_adaptive(linsys: Any, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    A = linsys.A
    A_abs = np.abs(A)
    n = linsys.nr_of_dims
    deltat = options['timeStep']
    taylorTermsGiven = 'taylorTerms' in options

    Apower = [A]
    Apower_abs = [A_abs]
    M = np.eye(n)
    Asum_pos = np.zeros((n, n))
    Asum_neg = np.zeros((n, n))

    eta = 1
    normAfro = []
    while True:
        Apower.append(Apower[eta - 1] @ A)
        Apower_abs.append(Apower_abs[eta - 1] @ A_abs)
        deltatbyfac_eta = deltat**eta / math.factorial(eta)
        M = M + Apower_abs[eta - 1] * deltatbyfac_eta

        if eta == 1:
            eta += 1
            continue

        exp1 = -eta / (eta - 1)
        exp2 = -1 / (eta - 1)
        factor = (eta**exp1 - eta**exp2) * deltatbyfac_eta

        Apos = np.zeros((n, n))
        Aneg = np.zeros((n, n))
        pos_ind = Apower[eta - 1] > 0
        neg_ind = Apower[eta - 1] < 0
        Apos[pos_ind] = Apower[eta - 1][pos_ind]
        Aneg[neg_ind] = Apower[eta - 1][neg_ind]

        Asum_pos = Asum_pos + factor * Aneg
        Asum_neg = Asum_neg + factor * Apos

        normAfro.append(np.linalg.norm(Asum_pos - Asum_neg, ord='fro'))

        if taylorTermsGiven:
            stopCondition = eta == options['taylorTerms']
        else:
            prev = normAfro[-2] if len(normAfro) > 1 else normAfro[-1]
            stopCondition = (not np.any(Apower[eta - 1])) or (1 - prev / normAfro[-1] < options['zetaTlin'])

        if stopCondition:
            W = np.abs(expm(A_abs * options['timeStep']) - M)
            E = IntervalMatrix(np.zeros_like(W), W)
            Asum = _interval_matrix_from_bounds(Asum_neg, Asum_pos)
            deltatbyfac = [deltat**k / math.factorial(k) for k in range(1, eta + 2)]
            options['taylorTerms'] = eta
            options['factor'] = deltatbyfac
            if len(normAfro) > 1:
                options['etalinFro'] = 1 - normAfro[-2] / normAfro[-1]
            else:
                options['etalinFro'] = 0.0
            break
        else:
            eta += 1
            if eta > 50:
                raise Exception('expmtie:notconverging')

    linsys.taylor.powers = Apower
    linsys.taylor.error = E
    linsys.taylor.F = Asum + E

    return linsys, options


def _aux_inputSolution(linsys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    if 'W' in params:
        params['W'] = linsys.E * params['W']
        Wcenter = params['W'].center()
        W = params['W'] + (-Wcenter)
    else:
        Wcenter = 0
        W = 0

    V = linsys.B * params['U'] + W
    vTrans = linsys.B * params['uTrans'] + Wcenter + linsys.c

    options['isRV'] = not V.representsa_('origin', np.finfo(float).eps)

    Apower = linsys.taylor.powers
    E = linsys.taylor.error
    r = options['timeStep']
    n = linsys.nr_of_dims
    factors = options['factor']

    if options['isRV']:
        Vsum = r * V
        Asum = r * np.eye(n)
        for i in range(options['taylorTerms']):
            Vsum = Vsum + Apower[i] * factors[i + 1] * V
            Asum = Asum + Apower[i] * factors[i + 1]
        try:
            inputSolV = Vsum + (E * r) * V
        except Exception:
            inputSolV = Vsum + (E * r) * V.interval()
    else:
        Asum = r * np.eye(n)
        for i in range(options['taylorTerms']):
            Asum = Asum + Apower[i] * factors[i + 1]
        inputSolV = np.zeros((n, 1))

    eAtInt = Asum + (E * r)
    inputSolVtrans = eAtInt * Zonotope(vTrans)

    if options.get('originContained', False):
        inputCorr = np.zeros((n, 1))
    else:
        linsys = _aux_inputTie(linsys, options)
        inputF = linsys.taylor.inputF
        inputCorr = inputF * Zonotope(vTrans)

    linsys.taylor.V = V
    linsys.taylor.RV = inputSolV
    linsys.taylor.Rtrans = inputSolVtrans
    linsys.taylor.inputCorr = inputCorr
    linsys.taylor.eAtInt = eAtInt

    return linsys, options


def _aux_inputTie(linsys: Any, options: Dict[str, Any]) -> Any:
    Apower = linsys.taylor.powers
    E = linsys.taylor.error
    taylorTerms = options['taylorTerms']
    r = options['timeStep']
    n = linsys.nr_of_dims

    Asum_pos = np.zeros((n, n))
    Asum_neg = np.zeros((n, n))

    for i in range(2, taylorTerms + 2):
        exp1 = -i / (i - 1)
        exp2 = -1 / (i - 1)
        factor = (i**exp1 - i**exp2) * options['factor'][i - 1]

        Apos = np.zeros((n, n))
        Aneg = np.zeros((n, n))
        pos_ind = Apower[i - 2] > 0
        neg_ind = Apower[i - 2] < 0
        Apos[pos_ind] = Apower[i - 2][pos_ind]
        Aneg[neg_ind] = Apower[i - 2][neg_ind]

        Asum_pos = Asum_pos + factor * Aneg
        Asum_neg = Asum_neg + factor * Apos

    Asum = _interval_matrix_from_bounds(Asum_neg, Asum_pos)
    Einput = E * r
    linsys.taylor.inputF = Asum + Einput
    return linsys


def _interval_matrix_from_bounds(inf_mat: np.ndarray, sup_mat: np.ndarray) -> IntervalMatrix:
    center = 0.5 * (inf_mat + sup_mat)
    delta = 0.5 * (sup_mat - inf_mat)
    return IntervalMatrix(center, delta)
