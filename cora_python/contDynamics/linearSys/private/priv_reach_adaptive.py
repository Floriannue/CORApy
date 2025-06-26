"""
priv_reach_adaptive - computes an outer approximation of the reachable set
for linear time-invariant systems given a maximum error in terms of
the Hausdorff distance to the exact reachable set; all internal
reachability settings are set automatically

Syntax:
    timeInt, timePoint, res, savedata = priv_reach_adaptive(linsys, params, options)

Inputs:
    linsys - linearSys object
    params - model parameters
    options - options for the computation of reachable sets

Outputs:
    timeInt - array of time-interval reachable / output sets
    timePoint - array of time-point reachable / output sets
    res - specifications verified (only if called from verify)
    savedata - data used for subsequent runs (only if called from verify)

Example:
    -

References:
    [1] M. Wetzlinger et al. "Fully automated verification of linear
        systems using inner-and outer-approximations of reachable sets",
        TAC, 2023.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 08-July-2021 (MATLAB)
Last update: 22-March-2022 (MATLAB)
             06-November-2024 (MATLAB, full refactor)
Python translation: 2025
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from cora_python.g.classes.linErrorBound import LinErrorBound
from cora_python.g.classes.verifyTime import VerifyTime
from cora_python.g.functions.verbose import verboseLog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.classes.taylorLinSys import TaylorLinSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from .priv_correctionMatrixState import priv_correctionMatrixState
from .priv_correctionMatrixInput import priv_correctionMatrixInput


def priv_reach_adaptive(linsys, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict, Dict, bool, Dict]:
    """
    Computes an outer approximation of the reachable set for linear time-invariant systems
    using adaptive parameter tuning
    
    Args:
        linsys: LinearSys object
        params: Model parameters
        options: Computation options
        
    Returns:
        Tuple of (timeInt, timePoint, res, savedata)
    """
    
    # initializations ---------------------------------------------------------
    
    # Handle modern Python specification format
    # Convert to internal format for compatibility with existing adaptive reachability code
    if 'specification' in params:
        # Modern Python approach: single specification object or list
        spec_list = [params['specification']] if not isinstance(params['specification'], list) else params['specification']
        # Convert internally but keep the modern API
        params['_internal_safeSet'], params['_internal_unsafeSet'] = aux_getSetsFromSpec(spec_list)
    elif 'spec' in params:
        # Legacy MATLAB-style format support
        params['_internal_safeSet'], params['_internal_unsafeSet'] = aux_getSetsFromSpec(params['spec'])
    
    # call from verify?
    if 'verify' not in options:
        options['verify'] = False
    
    # for safety, init taylor helper class if not there already
    if not hasattr(linsys, 'taylor') or linsys.taylor is None or not hasattr(linsys.taylor, 'computeField'):
        linsys.taylor = TaylorLinSys(linsys.A)
    
    # init struct which keeps sets and timeStep to avoid recomputations
    savedata = aux_initSavedata(options)
    
    # time intervals where specifications are not yet verified
    timeSpecUnsat, params['tFinal'] = aux_initSpecUnsat(params, options['verify'])
    
    # init step and time (shift so that start at t=0), and spec satisfaction
    k = 0
    t = 0.0
    res = True
    params['tFinal'] = params['tFinal'] - params['tStart']
    params['tStart'] = 0.0
    
    # initialize first time step by time horizon
    timeStep = params['tFinal']
    
    # initialize all variables for sets
    set_vars = aux_initSets(linsys, params, options['verify'])
    
    # init over-approximation error w.r.t the corresponding exact reachable sets
    Rcont_error = []
    Rcont_tp_error = []
    
    # initialize time-point and time-interval output sets and over-approximation error
    timeInt = {'set': [], 'time': [], 'error': []}
    timePoint = {'set': [], 'time': [], 'error': []}
    
    # rewrite equations in canonical form
    linsys, params = aux_canonicalForm(linsys, params)

    # init all auxiliary flags
    isU, G_U, isu, isuconst = aux_initFlags(linsys, params)
    
    # compute factor for scaling of Hausdorff distance from state space to output space
    if hasattr(linsys, 'C') and linsys.C is not None:
        errR2Y = np.linalg.norm(linsys.C, 2)
    else:
        # Default to identity matrix if no output matrix specified
        linsys.C = np.eye(linsys.A.shape[0])
        errR2Y = 1.0
    
    # scale maximum error so that it corresponds to R (C = 1 => errR2Y = 1)
    errs = LinErrorBound(options['error'] / errR2Y, params['tFinal'])
    
    # time step adaptation using approximation functions or bisection:
    # first step always bisection (more robust)
    errs.useApproxFun = False
    
    # pre-compute how much of the total to allocate to the reduction error
    errs, savedata = aux_errorboundreduction(errs, linsys.A, isU, G_U, options, savedata)
    
    # compute first output solution and corresponding error
    V_init = params.get('V', np.zeros((linsys.C.shape[0], 1))) if params.get('V') is not None else np.zeros((linsys.C.shape[0], 1))
    vTrans_init = params.get('vTrans', np.zeros((linsys.C.shape[0], 1))) if params.get('vTrans') is not None else np.zeros((linsys.C.shape[0], 1))
    
    # Ensure V_init and vTrans_init have the correct output dimension
    if V_init.shape[0] != linsys.C.shape[0]:
        V_init = np.zeros((linsys.C.shape[0], 1))
    if vTrans_init.shape[0] != linsys.C.shape[0]:
        vTrans_init = np.zeros((linsys.C.shape[0], 1))
    
    timePoint['set'].append(linsys.C @ params['R0'] + V_init + vTrans_init)
    timePoint['time'].append(params['tStart'])
    timePoint['error'].append(0.0)
    
    # -------------------------------------------------------------------------
    
    # main loop until the time is up
    fullcomp = []
    while params['tFinal'] - t > 1e-9:
        
        # update step counter
        k += 1
        
        # log information
        verboseLog(options.get('verbose', 0), k, t, params['tStart'], params['tFinal'])
        
        # update constant input vector (if necessary)
        params['uTrans'], params['vTrans'] = aux_uTrans_vTrans(
            t, params.get('uTransVec'), params.get('vTransVec'), params.get('tu'))
        
        # flag whether full computation required
        if timeSpecUnsat.numIntervals() == 0:
            maxTimeStepSpec = float('inf')
            fullcomp.append(True)
        else:
            maxTimeStepSpec, fullcomp_k = timeSpecUnsat.timeUntilSwitch(t)
            fullcomp.append(fullcomp_k)
        
        # determine startset for current step
        if fullcomp[k-1]:  # Python 0-based indexing adjustment
            set_vars['startset'] = set_vars['Hstartp']
        
        # compute maximum admissible time step size for current step
        maxTimeStep = aux_maxTimeStep(t, params.get('uTransVec'), params.get('tu'),
                                      params['tFinal'], maxTimeStepSpec)
        
        # initial guess for time step size of the current step
        timeStep = min(maxTimeStep, timeStep)
        
        # initialize current step struct and iteration counter
        k_iter = 0
        set_vars = aux_prepareSets(set_vars)
        
        # loop until both errors are satisfied
        timeStepIdx = []
        while True:
            
            # increase counter for time step sizes used in this step
            k_iter += 1
            
            # compute error bounds
            errs.nextBounds(timeStep, t, k-1, k_iter-1)  # Adjust for 0-based indexing
            
            # check if time step size was used before
            timeStepIdx.append(aux_checkTimeStep(timeStep, savedata))
            
            # compute errors for given time step size
            errs, set_vars, converged = aux_errors(
                linsys, k-1, k_iter-1, errs, fullcomp[k-1], isU, isuconst, G_U,
                params.get('uTrans'), set_vars, t, timeStep, timeStepIdx[k_iter-1], savedata)
            
            if not converged:
                # if computation has not converged, repeat with smaller time step
                if k_iter == 1:
                    timeStep = timeStep * 0.1
                    k_iter = 0
                    set_vars = aux_prepareSets(set_vars)
                    continue
                else:
                    # take sets for time step from previous iteration
                    k_iter_chosen = next((i for i, idx in enumerate(timeStepIdx) 
                                        if idx == timeStepIdx[k_iter-2]), k_iter-2)
                    break
            
            # if new time step size used, save auxiliary variables
            if (timeStepIdx[k_iter-1] == 0 or
                (timeStepIdx[k_iter-1] < len(savedata.get('fullcomp', [])) and
                 fullcomp[k-1] != savedata['fullcomp'][timeStepIdx[k_iter-1]])):
                timeStepIdx[k_iter-1], savedata = aux_savedata(
                    savedata, k-1, k_iter-1, fullcomp[k-1], isU, isuconst, set_vars, errs, timeStep)
            
            # init/update coefficients of approximation functions
            errs.updateCoefficientsApproxFun(k-1, k_iter-1, fullcomp[k-1])
            
            # check non-accumulating and accumulating error against bound
            errs.bound_acc_ok.append([])
            errs.bound_nonacc_ok.append([])
            while len(errs.bound_acc_ok) <= k-1:
                errs.bound_acc_ok.append([])
            while len(errs.bound_nonacc_ok) <= k-1:
                errs.bound_nonacc_ok.append([])
            while len(errs.bound_acc_ok[k-1]) <= k_iter-1:
                errs.bound_acc_ok[k-1].append(False)
            while len(errs.bound_nonacc_ok[k-1]) <= k_iter-1:
                errs.bound_nonacc_ok[k-1].append(False)
                
            errs.bound_acc_ok[k-1][k_iter-1] = (errs.step_acc[k-1][k_iter-1] < 
                                               errs.bound_acc[k-1][k_iter-1])
            errs.bound_nonacc_ok[k-1][k_iter-1] = (errs.seq_nonacc[k-1][k_iter-1] < 
                                                  errs.bound_rem[k-1][k_iter-1] or not fullcomp[k-1])
            
            # update bisect values
            errs.updateBisection(k-1, k_iter-1, isU, timeStep)
            
            # check whether suitable time step size found
            if (errs.bound_acc_ok[k-1][k_iter-1] and 
                (not fullcomp[k-1] or errs.bound_nonacc_ok[k-1][k_iter-1])):
                
                # suitable time step size found
                k_iter_chosen = k_iter - 1  # Adjust for 0-based indexing
                break
            
            # time step size too large -> estimate new time step size
            if errs.useApproxFun and k_iter > 1:
                # use approximation functions
                timeStep = errs.estimateTimeStepSize(t, k-1, k_iter-1, fullcomp[k-1],
                                                   timeStep, maxTimeStep, isU)
            else:
                # use bisection method
                timeStep = errs.estimateTimeStepSize(t, k-1, k_iter-1, fullcomp[k-1],
                                                   timeStep, maxTimeStep, isU)
            
            # adjust time step size to avoid recomputations
            timeStep, _ = aux_adjustTimeStep(k-1, timeStep, isU, maxTimeStep, savedata)
        
        # clean up: remove sets for other time step sizes
        set_vars = aux_postCleanUpSets(set_vars, k_iter_chosen, fullcomp[k-1], isU)
        
        # accumulate particular solution and reduce
        while len(errs.step_red) <= k-1:
            errs.step_red.append(0.0)
        set_vars, errs.step_red[k-1] = aux_reduce(linsys, t, set_vars, 
                                                 errs.bound_red[k-1][k_iter_chosen], isU)
        
        # accumulate accumulating error and reduction error, save non-acc error
        errs.accumulateErrors(k-1, k_iter_chosen)
        # clean up: values for other time step sizes no longer needed
        errs.removeRedundantValues(k-1, k_iter_chosen)
        
        # compute particular solution due to input vector
        if isu:
            # due to current method of computation, this induces no error
            if not isuconst or len(savedata.get('Pu', [])) <= timeStepIdx[k_iter_chosen]:
                set_vars['Pu'] = particularSolution_constant(linsys, params['uTrans'], timeStep, float('inf'))
                if isuconst:
                    if 'Pu' not in savedata:
                        savedata['Pu'] = []
                    while len(savedata['Pu']) <= timeStepIdx[k_iter_chosen]:
                        savedata['Pu'].append(None)
                    savedata['Pu'][timeStepIdx[k_iter_chosen]] = set_vars['Pu']
            else:
                # spare re-computation if time step size has been used before
                set_vars['Pu'] = savedata['Pu'][timeStepIdx[k_iter_chosen]]
            
            # propagate constant input solution
            eAdtk = linsys.taylor.computeField('eAdt', timeStep=timeStep)
            set_vars['Putotal'] = eAdtk @ set_vars['Putotal'] + set_vars['Pu']
        
        # propagate reachable sets
        eAdtk = linsys.taylor.computeField('eAdt', timeStep=timeStep)
        set_vars['Hstartp'] = eAdtk @ set_vars['startset']
        
        # add particular solutions and curvature error
        set_vars['Rset_tp'] = set_vars['Hstartp']
        if isU:
            set_vars['Rset_tp'] = set_vars['Rset_tp'] + set_vars['Putotal']
        if isu:
            set_vars['Rset_tp'] = set_vars['Rset_tp'] + set_vars['Pu']
        
        # compute time-interval solution
        set_vars['Rset_ti'] = set_vars['startset'].enclose(set_vars['Hstartp'])
        if isU:
            set_vars['Rset_ti'] = set_vars['Rset_ti'] + set_vars['Putotal']
        if isu:
            set_vars['Rset_ti'] = set_vars['Rset_ti'] + set_vars['Pu']
        
        # compute output sets
        V_term = params.get('V', np.zeros((linsys.C.shape[0], 1))) if params.get('V') is not None else np.zeros((linsys.C.shape[0], 1))
        vTrans_term = params.get('vTrans', np.zeros((linsys.C.shape[0], 1))) if params.get('vTrans') is not None else np.zeros((linsys.C.shape[0], 1))
        
        # gather over-approximation errors using fullErrors method
        Rcont_error, Rcont_tp_error = errs.fullErrors(k-1)  # k-1 for 0-based indexing
        
        timeInt['set'].append(linsys.C @ set_vars['Rset_ti'] + V_term + vTrans_term)
        timeInt['time'].append([t, t + timeStep])
        timeInt['error'].append(Rcont_error)
        
        timePoint['set'].append(linsys.C @ set_vars['Rset_tp'] + V_term + vTrans_term)
        timePoint['time'].append(t + timeStep)
        timePoint['error'].append(Rcont_tp_error)
        
        # increment time
        t += timeStep
        
        # adaptive time step size for next step
        if errs.useApproxFun:
            timeStep = errs.estimateTimeStepSize(t, k-1, k_iter_chosen, fullcomp[k-1], 
                                               timeStep, maxTimeStep, isU)
        
        # update time step adaptation method
        if k == 1:
            errs.useApproxFun = True
    
    return timeInt, timePoint, res, savedata


# --- Auxiliary functions (mirroring MATLAB structure) ---

def aux_initFlags(linsys, params: Dict[str, Any]) -> Tuple[bool, np.ndarray, bool, bool]:
    """Initialize auxiliary flags"""
    isU = 'U' in params and not params['U'].representsa_('origin')
    # G_U should be B * U.G to transform input generators to state space
    if isU and hasattr(params['U'], 'G'):
        G_U = linsys.B @ params['U'].G  # Transform input generators through B matrix
    else:
        G_U = np.zeros((params['R0'].dim(), 1))
    isu = 'uTrans' in params and np.any(params['uTrans'])
    isuconst = isu and ('uTransVec' not in params or params['uTransVec'].shape[1] == 1)
    return isU, G_U, isu, isuconst


def aux_initSets(linsys, params: Dict[str, Any], verify: bool) -> Dict[str, Any]:
    """Initialize all variables for sets"""
    return {
        'startset': params['R0'],
        'Hstartp': params['R0'],
        'Putotal': np.zeros((linsys.A.shape[0], 1)),
        'Pu': None,
        'Rset_tp': None,
        'Rset_ti': None
    }


def aux_prepareSets(set_vars: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize current step struct"""
    # Reset iteration-specific variables
    return set_vars


def aux_postCleanUpSets(set_vars: Dict[str, Any], k_iter: int, fullcomp: bool, isU: bool) -> Dict[str, Any]:
    """Clean up: remove sets for other time step sizes"""
    # In Python, we don't need explicit cleanup like MATLAB
    return set_vars


def aux_canonicalForm(linsys, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Convert system to canonical form by ensuring that all required parameters are present
    and properly initialized for the adaptive reachability computation
    """
    # Ensure system has required matrices first
    if not hasattr(linsys, 'B') or linsys.B is None:
        # No input matrix - create zero matrix
        linsys.B = np.zeros((linsys.A.shape[0], 1))
    
    if not hasattr(linsys, 'C') or linsys.C is None:
        # No output matrix - use identity (observe all states)
        linsys.C = np.eye(linsys.A.shape[0])
    
    # Now set output space dimensions for V and vTrans
    output_dim = linsys.C.shape[0]
    state_dim = linsys.A.shape[0]
    input_dim = linsys.B.shape[1]
    
    # Ensure required parameters exist with proper defaults
    if 'V' not in params:
        params['V'] = np.zeros((output_dim, 1))
    if 'vTrans' not in params:
        params['vTrans'] = np.zeros((output_dim, 1))
    if 'uTrans' not in params:
        params['uTrans'] = np.zeros((input_dim, 1))
    
    # Ensure input set exists
    if 'U' not in params:
        params['U'] = Zonotope(np.zeros((input_dim, 1)), np.zeros((input_dim, 1)))
    
    return linsys, params


def aux_errorboundreduction(errs: LinErrorBound, A: np.ndarray, isU: bool, G_U: np.ndarray, 
                           options: Dict[str, Any], savedata: Dict[str, Any]) -> Tuple[LinErrorBound, Dict[str, Any]]:
    """Compute the percentage of the total error allocated to the reduction error"""
    if not isU:
        # no reduction, entire error margin is available for nonacc errors
        errs.bound_red_max = 0.0
        savedata['reductionerrorpercentage'] = 0.0
        return errs, savedata
    elif 'reductionerrorpercentage' in options:
        # define error curve manually
        errs.bound_red_max = errs.emax * options['reductionerrorpercentage']
        savedata['reductionerrorpercentage'] = errs.bound_red_max / errs.emax
        return errs, savedata
    elif 'reductionerrorpercentage' in savedata:
        # reduction error defined by previous run (only verify)
        errs.bound_red_max = errs.emax * savedata['reductionerrorpercentage']
        return errs, savedata
    
    # post-step: vector over time for reduction error allocation
    errs.computeErrorBoundReduction(A, G_U)
    
    # save value
    savedata['reductionerrorpercentage'] = errs.bound_red_max / errs.emax
    return errs, savedata


def aux_initSavedata(options: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all fields in struct savedata"""
    savedata = {
        'timeStep': [],
        'fullcomp': [],
        'Pu': [],
        'GuTrans_center': [],
        'GuTrans_Gbox': [],
        'eG': [],
        'G_PU_zero': [],
        'G_PU_infty': [],
        'PU_A_sum_error': []
    }
    
    # Read savedata from previous run if available
    if 'savedata' in options:
        for key, value in options['savedata'].items():
            if key in savedata:
                savedata[key] = value
    
    return savedata


def aux_initSpecUnsat(params: Dict[str, Any], verify: bool) -> Tuple[VerifyTime, float]:
    """Time intervals where specifications are not yet verified"""
    timeSpecUnsat = VerifyTime()
    tFinal = params['tFinal']
    
    if verify:
        times = []
        
        # Handle internal specification format (converted from modern Python specs)
        if '_internal_safeSet' in params:
            for safe_set in params['_internal_safeSet']:
                if 'time' in safe_set and safe_set['time'] is not None:
                    # Convert time to VerifyTime if needed
                    if hasattr(safe_set['time'], 'bounds'):
                        times.append(safe_set['time'])
                    else:
                        # Assume it's interval format
                        times.append(VerifyTime(safe_set['time']))
        
        if '_internal_unsafeSet' in params:
            for unsafe_set in params['_internal_unsafeSet']:
                if 'time' in unsafe_set and unsafe_set['time'] is not None:
                    # Convert time to VerifyTime if needed
                    if hasattr(unsafe_set['time'], 'bounds'):
                        times.append(unsafe_set['time'])
                    else:
                        # Assume it's interval format
                        times.append(VerifyTime(unsafe_set['time']))
        
        # Merge all time intervals
        if times:
            # Collect all bounds and shift by start time
            all_bounds = []
            for t in times:
                if hasattr(t, 'bounds') and t.bounds.size > 0:
                    all_bounds.extend(t.bounds.tolist())
            
            if all_bounds:
                # Shift by start time (subtract tStart to normalize to 0)
                shifted_bounds = np.array(all_bounds) - params['tStart']
                timeSpecUnsat = VerifyTime(shifted_bounds)
                tFinal = timeSpecUnsat.finalTime() + params['tStart']
    
    return timeSpecUnsat, tFinal


def aux_uTrans_vTrans(t: float, uTransVec, vTransVec, tu) -> Tuple[Any, Any]:
    """Update constant input vector (if necessary)"""
    if uTransVec is not None and uTransVec.shape[1] > 1:
        # Find current time index
        idx = np.searchsorted(tu, t, side='right') - 1
        idx = max(0, min(idx, uTransVec.shape[1] - 1))
        uTrans = uTransVec[:, idx]
    else:
        uTrans = uTransVec[:, 0] if uTransVec is not None else None
    
    if vTransVec is not None and vTransVec.shape[1] > 1:
        # Find current time index
        idx = np.searchsorted(tu, t, side='right') - 1
        idx = max(0, min(idx, vTransVec.shape[1] - 1))
        vTrans = vTransVec[:, idx]
    else:
        vTrans = vTransVec[:, 0] if vTransVec is not None else None
    
    return uTrans, vTrans


def aux_maxTimeStep(t: float, uTransVec, tu, tFinal: float, maxTimeStepSpec: float) -> float:
    """Maximum admissible value for time step size in the current step"""
    # default: non time-varying input center
    maxTimeStep = tFinal - t
    
    # time-varying input center
    if uTransVec is not None and uTransVec.shape[1] > 1:
        # add 1e-12 for numerical stability
        idxNextSwitch = np.where(tu > t + 1e-12)[0]
        if len(idxNextSwitch) > 0:
            maxTimeStep = tu[idxNextSwitch[0]] - t
    
    # change in specSAT boolean cannot be crossed
    maxTimeStep = min(maxTimeStep, maxTimeStepSpec)
    
    return maxTimeStep


def aux_checkTimeStep(timeStep: float, savedata: Dict[str, Any]) -> int:
    """Check whether auxiliary sets have already been computed for this time step size"""
    if 'timeStep' not in savedata or len(savedata['timeStep']) == 0:
        return 0
    
    indices = np.where(np.abs(np.array(savedata['timeStep']) - timeStep) < 1e-9)[0]
    return indices[0] + 1 if len(indices) > 0 else 0  # MATLAB uses 1-based indexing


def aux_adjustTimeStep(k: int, timeStep: float, isU: bool, maxTimeStep: float, 
                      savedata: Dict[str, Any]) -> Tuple[float, int]:
    """Adjust the predicted time step size to avoid recomputations"""
    if k == 0:
        factor = 0.95
    elif isU:
        factor = 0.75
    else:
        factor = 0.80
    
    # eligible time step sizes
    if 'timeStep' not in savedata or len(savedata['timeStep']) == 0:
        return min(timeStep, maxTimeStep), 0
    
    timeSteps = np.array(savedata['timeStep'])
    idx_adm = (timeSteps < timeStep) & (timeSteps > factor * timeStep)
    
    if not np.any(idx_adm):
        timeStepIdx = 0
        # if proposed time step size between not largest or smallest until now
        if not (np.all(timeStep >= timeSteps) or np.all(timeStep <= timeSteps)):
            timeStep_ub = np.min(timeSteps[timeSteps >= timeStep])
            timeStep_lb = np.max(timeSteps[timeSteps <= timeStep])
            timeStep = timeStep_lb + 0.5 * (timeStep_ub - timeStep_lb)
    else:
        # read time step and corresponding index from savedata
        timeStep = np.max(timeSteps[idx_adm])
        timeStepIdx = np.where(idx_adm)[0][-1] + 1  # MATLAB uses 1-based indexing
    
    # ensure that maximum time step size not exceeded
    if timeStep > maxTimeStep:
        timeStep = maxTimeStep
        exact_matches = np.where(np.abs(timeSteps - timeStep) < 1e-14)[0]
        timeStepIdx = max(0, exact_matches[0] + 1 if len(exact_matches) > 0 else 0)
    
    return timeStep, timeStepIdx


def aux_errors(linsys, k: int, k_iter: int, errs: LinErrorBound, fullcomp: bool,
               isU: bool, isuconst: bool, G_U: np.ndarray, u, set_vars: Dict[str, Any],
               t: float, timeStep: float, timeStepIdx: int, savedata: Dict[str, Any]) -> Tuple[LinErrorBound, Dict[str, Any], bool]:
    """
    Compute all errors for the given time step size, that is,
    accumulating:     e.idv_PUtkplus1
    non-accumulating: e.idv_linComb, e.idv_PUtauk, e.idv_F, e.idv_G
    and reduction:    e.step_red (set to 0 as reduction is performed later)
    furthermore, auxiliary variables such as F, G are computed, as well
    as the set PU which is closely related to its respective error
    """
    
    # compute exponential matrix
    eAdtk = linsys.taylor.computeField('eAdt', timeStep=timeStep)
    eAtk = linsys.taylor.computeField('eAdt', timeStep=t)
    converged = True
    
    # Ensure error lists are properly sized
    while len(errs.idv_PUtkplus1) <= k:
        errs.idv_PUtkplus1.append([])
    while len(errs.idv_PUtkplus1[k]) <= k_iter:
        errs.idv_PUtkplus1[k].append(0.0)
    
    # Ensure timeSteps is properly initialized
    while len(errs.timeSteps) <= k:
        errs.timeSteps.append([])
    while len(errs.timeSteps[k]) <= k_iter:
        errs.timeSteps[k].append(0.0)
    
    # Set the current time step
    errs.timeSteps[k][k_iter] = timeStep
    
    # compute accumulating error - consists of eps_PU_tkplus1
    if not isU:
        set_vars['eAtkPU'] = np.zeros((linsys.A.shape[0], 1))
        errs.idv_PUtkplus1[k][k_iter] = 0.0
    elif timeStepIdx == 0:
        converged, G_PU_zero, G_PU_infty, PU_A_sum_error = aux_PU(linsys, G_U, timeStep)
        if not converged:
            return errs, set_vars, False
        
        # Store in set_vars
        if 'G_PU_zero' not in set_vars:
            set_vars['G_PU_zero'] = []
        if 'G_PU_infty' not in set_vars:
            set_vars['G_PU_infty'] = []
        if 'PU_A_sum_error' not in set_vars:
            set_vars['PU_A_sum_error'] = []
        
        while len(set_vars['G_PU_zero']) <= k_iter:
            set_vars['G_PU_zero'].append(None)
            set_vars['G_PU_infty'].append(None)
            set_vars['PU_A_sum_error'].append(None)
        
        set_vars['G_PU_zero'][k_iter] = G_PU_zero
        set_vars['G_PU_infty'][k_iter] = G_PU_infty
        set_vars['PU_A_sum_error'][k_iter] = PU_A_sum_error
        
        # over-approximation of the Hausdorff distance
        errs.idv_PUtkplus1[k][k_iter] = (
            np.linalg.norm(np.sum(np.abs((eAtk @ PU_A_sum_error) @ G_U), axis=1)) +
            np.linalg.norm(np.sum(np.abs(eAtk @ G_PU_infty), axis=1))
        )
    else:
        # re-use values from memory
        set_vars['G_PU_infty'][k_iter] = savedata['G_PU_infty'][timeStepIdx - 1]
        set_vars['PU_A_sum_error'][k_iter] = savedata['PU_A_sum_error'][timeStepIdx - 1]
        errs.idv_PUtkplus1[k][k_iter] = (
            np.linalg.norm(np.sum(np.abs((eAtk @ set_vars['PU_A_sum_error'][k_iter]) @ G_U), axis=1)) +
            np.linalg.norm(np.sum(np.abs(eAtk @ set_vars['G_PU_infty'][k_iter]), axis=1))
        )
        set_vars['G_PU_zero'][k_iter] = timeStep * G_U
    
    # Initialize other error lists
    for attr in ['idv_F', 'idv_G', 'idv_linComb', 'idv_PUtauk', 'seq_nonacc']:
        if not hasattr(errs, attr):
            setattr(errs, attr, [])
        while len(getattr(errs, attr)) <= k:
            getattr(errs, attr).append([])
        while len(getattr(errs, attr)[k]) <= k_iter:
            getattr(errs, attr)[k].append(np.nan)
    
    if fullcomp:
        # linear combination error: eps_linComb
        errs.idv_linComb[k][k_iter] = errs.compute_eps_linComb(eAdtk, set_vars['startset'])
        
        # curvature error from correction matrix for the state
        try:
            F = priv_correctionMatrixState(linsys, timeStep, float('inf'))
        except Exception as e:
            if 'notConverged' in str(e):
                return errs, set_vars, False
            else:
                raise e
        
        errs.idv_F[k][k_iter], boxFstartset_center, boxFstartset_Gbox = errs.compute_eps_F(F, set_vars['startset'])
        
        # curvature error from correction matrix for the input
        if 'GuTrans_center' not in set_vars:
            set_vars['GuTrans_center'] = []
            set_vars['GuTrans_Gbox'] = []
        while len(set_vars['GuTrans_center']) <= k_iter:
            set_vars['GuTrans_center'].append(np.zeros((linsys.A.shape[0], 1)))
            set_vars['GuTrans_Gbox'].append(np.zeros((linsys.A.shape[0], 1)))
        
        if u is None or not np.any(u):
            errs.idv_G[k][k_iter] = 0.0
        else:
            try:
                G = priv_correctionMatrixInput(linsys, timeStep, float('inf'))
            except Exception as e:
                if 'notConverged' in str(e):
                    return errs, set_vars, False
                else:
                    raise e
            
            # load results from savedata
            if (isuconst and timeStepIdx > 0 and 
                len(savedata.get('fullcomp', [])) > timeStepIdx - 1 and
                savedata['fullcomp'][timeStepIdx - 1]):
                set_vars['GuTrans_center'][k_iter] = savedata['GuTrans_center'][timeStepIdx - 1]
                set_vars['GuTrans_Gbox'][k_iter] = savedata['GuTrans_Gbox'][timeStepIdx - 1]
                errs.idv_G[k][k_iter] = savedata['eG'][timeStepIdx - 1]
            else:
                errs.idv_G[k][k_iter], set_vars['GuTrans_center'][k_iter], set_vars['GuTrans_Gbox'][k_iter] = errs.compute_eps_G(G, u)
        
        # err(e^At PU)
        if not isU:
            errs.idv_PUtauk[k][k_iter] = 0.0
        else:
            errs.idv_PUtauk[k][k_iter] = (
                np.linalg.norm(np.sum(np.abs(eAtk @ set_vars['G_PU_zero'][k_iter]), axis=1) +
                              np.sum(np.abs(eAtk @ set_vars['G_PU_infty'][k_iter]), axis=1))
            )
        
        # total non-accumulating error
        errs.seq_nonacc[k][k_iter] = (errs.idv_linComb[k][k_iter] +
                                     errs.idv_PUtauk[k][k_iter] +
                                     errs.idv_F[k][k_iter] +
                                     errs.idv_G[k][k_iter])
    
    # total accumulating error (contribution of current step)
    if not hasattr(errs, 'step_acc'):
        errs.step_acc = []
    while len(errs.step_acc) <= k:
        errs.step_acc.append([])
    while len(errs.step_acc[k]) <= k_iter:
        errs.step_acc[k].append(0.0)
    
    errs.step_acc[k][k_iter] = errs.idv_PUtkplus1[k][k_iter]
    
    return errs, set_vars, True


def aux_PU(linsys, G_U: np.ndarray, timeStep: float) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computation of the particular solution due to the uncertain input set U
    as well as the contained over-approximation error with respect to the
    exact particular solution; using a Taylor series where the truncation
    order is increased until the additional values are so small that the
    stored number (finite precision!) does not change anymore
    """
    # initialize particular solution (only generator matrices)
    G_PU_zero = timeStep * G_U
    PU_diag = np.sum(np.abs(G_PU_zero), axis=1)
    
    # initialize errors
    G_PU_infty = np.zeros((linsys.A.shape[0], 0))
    A_sum_error = np.zeros(linsys.A.shape)
    
    eta = 1
    
    # loop until floating-point precision
    conv = True
    while True:
        # Get A^eta and dt^eta/eta!
        Apower_mm = linsys.taylor.computeField('Apower', ithpower=eta)
        dtoverfac = (timeStep ** (eta + 1)) / math.factorial(eta + 1)
        
        # additional term (only matrix)
        addTerm = Apower_mm * dtoverfac
        addG_PU = addTerm @ G_U
        addG_PU_diag = np.sum(np.abs(addG_PU), axis=1)
        
        # safety check
        if eta == 75:
            conv = False
            return conv, G_PU_zero, G_PU_infty, A_sum_error
        
        # check if floating-point precision reached
        if np.all(np.abs(addG_PU_diag) <= np.finfo(float).eps * np.abs(PU_diag)):
            break
        
        # add term to simplified value for convergence
        PU_diag = PU_diag + addG_PU_diag
        
        # error terms
        A_sum_error = A_sum_error + addTerm
        G_PU_infty = np.hstack([G_PU_infty, addG_PU])
        
        # increment eta
        eta += 1
    
    return conv, G_PU_zero, G_PU_infty, A_sum_error


def aux_reduce(linsys, t: float, set_vars: Dict[str, Any], err_bound_red_k: float, isU: bool) -> Tuple[Dict[str, Any], float]:
    """
    Function which simultaneously propagates and reduces the particular solution due to the uncertainty U
    
    This function performs order reduction on the particular solution to keep the computational
    complexity manageable while tracking the reduction error.
    """
    if not isU or 'PUtotal' not in set_vars:
        return set_vars, 0.0
    
    # Get current particular solution
    PUtotal = set_vars['PUtotal']
    
    # If PUtotal is a zonotope, perform reduction
    if hasattr(PUtotal, 'reduce'):
        # Compute current order
        if hasattr(PUtotal, 'G'):
            current_order = PUtotal.G.shape[1] / PUtotal.dim()
            max_order = 50  # Maximum allowed order
            
            if current_order > max_order:
                # Perform reduction
                PUtotal_reduced = PUtotal.reduce('girard', max_order)
                
                # Compute reduction error (Hausdorff distance)
                if hasattr(PUtotal, 'hausdorffDistance'):
                    err_red_k = PUtotal.hausdorffDistance(PUtotal_reduced)
                else:
                    # Approximate reduction error
                    if hasattr(PUtotal, 'G') and hasattr(PUtotal_reduced, 'G'):
                        G_orig = PUtotal.G
                        G_red = PUtotal_reduced.G
                        # Estimate error as norm of removed generators
                        if G_orig.shape[1] > G_red.shape[1]:
                            removed_gens = G_orig[:, G_red.shape[1]:]
                            err_red_k = np.linalg.norm(np.sum(np.abs(removed_gens), axis=1))
                        else:
                            err_red_k = 0.0
                    else:
                        err_red_k = 0.0
                
                # Ensure error is within bound
                if err_red_k <= err_bound_red_k:
                    set_vars['PUtotal'] = PUtotal_reduced
                else:
                    # Reduction would violate error bound - keep original
                    err_red_k = 0.0
            else:
                err_red_k = 0.0
        else:
            err_red_k = 0.0
    else:
        # Not a reducible set
        err_red_k = 0.0
    
    return set_vars, err_red_k


def aux_savedata(savedata: Dict[str, Any], k: int, k_iter: int, fullcomp: bool,
                 isU: bool, isuconst: bool, set_vars: Dict[str, Any], 
                 errs: LinErrorBound, timeStep: float) -> Tuple[int, Dict[str, Any]]:
    """Save data into struct to avoid recomputations in subsequent steps"""
    # Find existing timeStepIdx or create new one
    timeStepIdx = None
    for i, ts in enumerate(savedata['timeStep']):
        if abs(ts - timeStep) < 1e-14:
            timeStepIdx = i + 1  # MATLAB uses 1-based indexing
            break
    
    if timeStepIdx is None:
        timeStepIdx = len(savedata['timeStep']) + 1
        savedata['timeStep'].append(timeStep)
        for key in ['fullcomp', 'Pu', 'GuTrans_center', 'GuTrans_Gbox', 'eG', 
                   'G_PU_zero', 'G_PU_infty', 'PU_A_sum_error']:
            if key not in savedata:
                savedata[key] = []
            savedata[key].append(None)
    else:
        # Update existing entry
        savedata['timeStep'][timeStepIdx - 1] = timeStep
    
    # Save values
    savedata['fullcomp'][timeStepIdx - 1] = fullcomp
    
    if isuconst:
        if fullcomp:
            savedata['GuTrans_center'][timeStepIdx - 1] = set_vars.get('GuTrans_center', [None] * (k_iter + 1))[k_iter]
            savedata['GuTrans_Gbox'][timeStepIdx - 1] = set_vars.get('GuTrans_Gbox', [None] * (k_iter + 1))[k_iter]
            savedata['eG'][timeStepIdx - 1] = errs.idv_G[k][k_iter] if len(errs.idv_G) > k and len(errs.idv_G[k]) > k_iter else 0.0
        else:
            savedata['GuTrans_center'][timeStepIdx - 1] = None
            savedata['GuTrans_Gbox'][timeStepIdx - 1] = None
            savedata['eG'][timeStepIdx - 1] = None
    
    # particular solution
    if isU:
        savedata['G_PU_zero'][timeStepIdx - 1] = set_vars.get('G_PU_zero', [None] * (k_iter + 1))[k_iter]
        savedata['G_PU_infty'][timeStepIdx - 1] = set_vars.get('G_PU_infty', [None] * (k_iter + 1))[k_iter]
        savedata['PU_A_sum_error'][timeStepIdx - 1] = set_vars.get('PU_A_sum_error', [None] * (k_iter + 1))[k_iter]
    else:
        savedata['G_PU_zero'][timeStepIdx - 1] = None
        savedata['G_PU_infty'][timeStepIdx - 1] = None
        savedata['PU_A_sum_error'][timeStepIdx - 1] = None
    
    return timeStepIdx, savedata


# Helper functions that would need proper implementation
# Note: _representsa_origin function removed - use set.representsa_('origin') method directly


# Note: enclose function removed - use set1.enclose(set2) method directly


def particularSolution_constant(linsys, u, timeStep: float, truncationOrder: float):
    """Compute particular solution for constant input"""
    return linsys.particularSolution_constant(u, timeStep, truncationOrder)[0]  # Return only Ptp


# Note: compute_eps_* functions have been moved to LinErrorBound class methods 

def aux_getSetsFromSpec(spec_list):
    """
    Extract polytopic safe sets and unsafe sets from the specifications
    
    Args:
        spec_list: List of specification objects
        
    Returns:
        tuple: (safeSet, unsafeSet) cell arrays with .set, .time properties
    """
    
    # All specifications must be of type 'safeSet' or 'unsafeSet'
    for spec in spec_list:
        if spec.type not in ['safeSet', 'unsafeSet']:
            raise CORAerror('CORA:notSupported',
                           "Only specification types 'safeSet' and 'unsafeSet' are supported.")
    
    # Pre-allocate length, truncate at the end
    num_spec = len(spec_list)
    safe_set = [None] * num_spec
    unsafe_set = [None] * num_spec
    idx_safe_set = 0
    idx_unsafe_set = 0
    
    for spec in spec_list:
        # Convert set to a polytope and normalize constraint vectors
        if hasattr(spec.set, 'polytope'):
            P_norm = spec.set.polytope()
        else:
            P_norm = Polytope(spec.set)
        
        # TODO: Add normalizeConstraints equivalent if needed
        
        if spec.type == 'safeSet':
            safe_set[idx_safe_set] = {
                'set': P_norm,
                'time': spec.time
            }
            idx_safe_set += 1
            
        elif spec.type == 'unsafeSet':
            # TODO: Check if P_norm represents a halfspace and convert to safe set if so
            # For now, just add to unsafe set
            unsafe_set[idx_unsafe_set] = {
                'set': P_norm,
                'time': spec.time,
                'isBounded': P_norm.isBounded() if hasattr(P_norm, 'isBounded') else True
            }
            
            # Pre-compute interval enclosure if bounded
            if unsafe_set[idx_unsafe_set]['isBounded']:
                try:
                    unsafe_set[idx_unsafe_set]['int'] = P_norm.interval()
                except Exception:
                    # Fallback: no interval precomputation
                    unsafe_set[idx_unsafe_set]['int'] = None
            
            idx_unsafe_set += 1
    
    # Truncate lists according to collected specs
    safe_set = safe_set[:idx_safe_set]
    unsafe_set = unsafe_set[:idx_unsafe_set]
    
    return safe_set, unsafe_set 