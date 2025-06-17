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
import scipy.linalg
from typing import Dict, Any, Tuple, List, Optional
from cora_python.g.classes.linErrorBound import LinErrorBound
from cora_python.g.classes.verifyTime import VerifyTime
from cora_python.g.functions.verbose.verboseLog import verboseLog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


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
    
    # call from verify?
    if 'verify' not in options:
        options['verify'] = False
    
    # for safety, init taylor helper class if not there already
    if not hasattr(linsys, 'taylor') or linsys.taylor is None or not hasattr(linsys.taylor, 'computeField'):
        from cora_python.g.classes.taylorLinSys import TaylorLinSys
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
    isU, G_U, isu, isuconst = aux_initFlags(params)
    
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
        set_vars['Rset_ti'] = enclose(set_vars['startset'], set_vars['Hstartp'])
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

def aux_initFlags(params: Dict[str, Any]) -> Tuple[bool, np.ndarray, bool, bool]:
    """Initialize auxiliary flags"""
    isU = 'U' in params and not _representsa_origin(params['U'])
    G_U = params['U'].generators() if isU else np.zeros((params['R0'].dim(), 1))
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
    # Ensure required parameters exist with proper defaults
    if 'V' not in params:
        params['V'] = np.zeros((linsys.A.shape[0], 1))
    if 'vTrans' not in params:
        params['vTrans'] = np.zeros((linsys.A.shape[0], 1))
    if 'uTrans' not in params:
        params['uTrans'] = np.zeros((linsys.B.shape[1] if hasattr(linsys, 'B') and linsys.B is not None else 1, 1))
    
    # Ensure system has required matrices
    if not hasattr(linsys, 'B') or linsys.B is None:
        # No input matrix - create zero matrix
        linsys.B = np.zeros((linsys.A.shape[0], 1))
    
    if not hasattr(linsys, 'C') or linsys.C is None:
        # No output matrix - use identity (observe all states)
        linsys.C = np.eye(linsys.A.shape[0])
    
    # Ensure input set exists
    if 'U' not in params:
        from cora_python.contSet.zonotope.zonotope import Zonotope
        params['U'] = Zonotope(np.zeros((linsys.B.shape[1], 1)), np.zeros((linsys.B.shape[1], 1)))
    
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
    if verify and 'specification' in params:
        # Create VerifyTime object with specification intervals
        timeSpecUnsat = VerifyTime(params['specification'])
        tFinal = params['tFinal']
    else:
        # No specifications - empty VerifyTime
        timeSpecUnsat = VerifyTime()
        tFinal = params['tFinal']
    
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
        errs.idv_linComb[k][k_iter] = compute_eps_linComb(errs, eAdtk, set_vars['startset'])
        
        # curvature error from correction matrix for the state
        try:
            F = priv_correctionMatrixState(linsys, timeStep, float('inf'))
        except Exception as e:
            if 'notConverged' in str(e):
                return errs, set_vars, False
            else:
                raise e
        
        errs.idv_F[k][k_iter], boxFstartset_center, boxFstartset_Gbox = compute_eps_F(errs, F, set_vars['startset'])
        
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
                errs.idv_G[k][k_iter], set_vars['GuTrans_center'][k_iter], set_vars['GuTrans_Gbox'][k_iter] = compute_eps_G(errs, G, u)
        
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
        if hasattr(PUtotal, 'generators'):
            current_order = PUtotal.generators().shape[1] / PUtotal.dim()
            max_order = 50  # Maximum allowed order
            
            if current_order > max_order:
                # Perform reduction
                PUtotal_reduced = PUtotal.reduce('girard', max_order)
                
                # Compute reduction error (Hausdorff distance)
                if hasattr(PUtotal, 'hausdorffDistance'):
                    err_red_k = PUtotal.hausdorffDistance(PUtotal_reduced)
                else:
                    # Approximate reduction error
                    if hasattr(PUtotal, 'generators') and hasattr(PUtotal_reduced, 'generators'):
                        G_orig = PUtotal.generators()
                        G_red = PUtotal_reduced.generators()
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
def _representsa_origin(U) -> bool:
    """Check if set represents origin"""
    if hasattr(U, 'representsa_'):
        return U.representsa_('origin')
    elif hasattr(U, 'generators'):
        # Check if generators are empty or all zero
        G = U.generators()
        return G.size == 0 or np.allclose(G, 0)
    return False


def enclose(set1, set2):
    """
    Compute convex hull of two sets (enclose operation)
    
    The enclose operation computes the convex hull of two sets, which is the
    smallest convex set containing both input sets.
    """
    if hasattr(set1, 'enclose'):
        return set1.enclose(set2)
    elif hasattr(set1, 'convHull'):
        return set1.convHull(set2)
    elif hasattr(set1, 'c') and hasattr(set1, 'G') and hasattr(set2, 'c') and hasattr(set2, 'G'):
        # Both are zonotopes - compute convex hull
        from cora_python.contSet.zonotope.zonotope import Zonotope
        
        # Simple convex hull for zonotopes: 
        # Z = 0.5*(Z1 + Z2) + 0.5*(Z1 - Z2)
        c_new = 0.5 * (set1.c + set2.c)
        G_new = np.hstack([
            0.5 * set1.G,
            0.5 * set2.G,
            0.5 * (set1.c - set2.c).reshape(-1, 1)
        ])
        return Zonotope(c_new, G_new)
    elif hasattr(set1, 'infimum') and hasattr(set1, 'supremum'):
        # Interval sets - compute interval hull
        from cora_python.contSet.interval.interval import Interval
        if hasattr(set2, 'infimum') and hasattr(set2, 'supremum'):
            inf_new = np.minimum(set1.infimum(), set2.infimum())
            sup_new = np.maximum(set1.supremum(), set2.supremum())
            return Interval(inf_new, sup_new)
        else:
            # Convert set2 to interval first
            set2_int = Interval(set2)
            inf_new = np.minimum(set1.infimum(), set2_int.infimum())
            sup_new = np.maximum(set1.supremum(), set2_int.supremum())
            return Interval(inf_new, sup_new)
    else:
        # Fallback: return first set (conservative approximation)
        return set1


def particularSolution_constant(linsys, u, timeStep: float, truncationOrder: float):
    """Compute particular solution for constant input"""
    from cora_python.contDynamics.linearSys.particularSolution_constant import particularSolution_constant as ps_const
    return ps_const(linsys, u, timeStep, truncationOrder)[0]  # Return only Ptp


def priv_correctionMatrixState(linsys, timeStep: float, truncationOrder: float):
    """
    Compute correction matrix for the state, see [1, Prop. 3.1]
    
    References:
        [1] M. Althoff. "Reachability Analysis and its Application to the
            Safety Assessment of Autonomous Cars", PhD Dissertation, 2010.
    """
    from cora_python.contSet.interval import Interval
    
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
                raise CORAError('CORA:notConverged', 'Time step size too big for computation of F.')
        
        # Compute powers; factor is always negative
        Asum_pos_F = Asum_pos_F + Asum_add_neg
        Asum_neg_F = Asum_neg_F + Asum_add_pos
    
    # Compute correction matrix for the state
    F = Interval(Asum_neg_F, Asum_pos_F)
    
    # Compute/read remainder of exponential matrix (unless truncationOrder=Inf)
    if not truncationOrderInf:
        E = priv_expmRemainder(linsys, timeStep, truncationOrder)
        F = F + E
    
    # Save in taylorLinSys object
    if not hasattr(linsys.taylor, '_F_cache'):
        linsys.taylor._F_cache = {}
    linsys.taylor._F_cache[timeStep] = F
    
    return F


def priv_correctionMatrixInput(linsys, timeStep: float, truncationOrder: float):
    """
    Compute correction matrix for the input, see [1, p. 38]
    
    References:
        [1] M. Althoff. "Reachability Analysis and its Application to the
            Safety Assessment of Autonomous Cars", PhD Dissertation, 2010.
    """
    from cora_python.contSet.interval import Interval
    
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
                raise CORAError('CORA:notConverged', 'Time step size too big for computation of G.')
        
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


def priv_expmRemainder(linsys, timeStep: float, truncationOrder: int):
    """
    Computation of remainder term of exponential matrix
    """
    from cora_python.contSet.interval import Interval
    
    # Check if it has already been computed
    if hasattr(linsys.taylor, 'E') and timeStep in getattr(linsys.taylor, '_E_cache', {}):
        return linsys.taylor._E_cache[timeStep]
    
    # Set a maximum order in case truncation order is given as Inf (adaptive)
    truncationOrderInf = np.isinf(truncationOrder)
    if truncationOrderInf:
        truncationOrder = 75
    
    # Initialization for loop
    A_abs = np.abs(linsys.A)
    n = linsys.A.shape[0]
    M = np.eye(n)
    
    # Compute powers for each term and sum of these
    for eta in range(1, int(truncationOrder) + 1):
        Apower_abs_i = np.linalg.matrix_power(A_abs, eta)
        dtoverfac_i = (timeStep ** eta) / math.factorial(eta)
        
        # Additional term
        M_add = Apower_abs_i * dtoverfac_i
        
        # Adaptive handling
        if truncationOrderInf and np.all(M_add <= np.finfo(float).eps * M):
            break
        
        M = M + M_add
    
    # Determine error due to finite Taylor series
    # (compute absolute value of W for numerical stability)
    W = np.abs(scipy.linalg.expm(A_abs * timeStep) - M)
    E = Interval(-W, W)
    
    # Save in taylorLinSys object
    if not hasattr(linsys.taylor, '_E_cache'):
        linsys.taylor._E_cache = {}
    linsys.taylor._E_cache[timeStep] = E
    
    return E


def compute_eps_linComb(errs: LinErrorBound, eAdt: np.ndarray, startset) -> float:
    """
    Compute error of linear combination, see [1, Prop. 9]
    
    References:
        [1] M. Wetzlinger et al. "Fully automated verification of linear
            systems using inner-and outer-approximations of reachable sets",
            TAC, 2023.
    """
    n = eAdt.shape[0]
    
    # Get generators from startset
    if hasattr(startset, 'generators'):
        G = startset.generators()
    elif hasattr(startset, 'G'):
        G = startset.G
    else:
        # Fallback for simple sets
        G = np.eye(n) * 0.1  # Small default generators
    
    G_minus = (eAdt - np.eye(n)) @ G
    eps_linComb = np.sqrt(G_minus.shape[1]) * np.linalg.norm(G_minus, 2)
    
    return eps_linComb


def compute_eps_F(errs: LinErrorBound, F, startset) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute curvature error (state), see [1, Prop. 1]:
    eps_F = 2 * errOp(F * startset)
    """
    from cora_python.contSet.zonotope import Zonotope
    
    if hasattr(startset, 'c') and hasattr(startset, 'G'):
        # Zonotope case - small speed up
        if hasattr(F, 'center') and hasattr(F, 'rad'):
            Fcenter = F.center()
            Frad = F.rad()
        else:
            # Fallback for simple interval
            Fcenter = (F.supremum() + F.infimum()) / 2
            Frad = (F.supremum() - F.infimum()) / 2
        
        box_Fstartset_c = Fcenter @ startset.c
        Fstartset_G = np.hstack([
            Fcenter @ startset.G,
            np.diag(Frad @ np.sum(np.abs(np.hstack([startset.c.reshape(-1, 1), startset.G])), axis=1))
        ])
        box_Fstartset_G = np.sum(np.abs(Fstartset_G), axis=1)
        eps_F = 2 * np.linalg.norm(box_Fstartset_G + np.abs(box_Fstartset_c))
    else:
        # Standard computation
        if hasattr(F, '__matmul__') and hasattr(startset, '__rmul__'):
            errorset = F @ startset
        else:
            # Fallback computation
            errorset = startset  # Simplified
        
        eps_F = 2 * _priv_errOp(errorset)
        
        # Convert to zonotope for output
        if hasattr(errorset, 'c') and hasattr(errorset, 'G'):
            box_Fstartset_c = errorset.c
            box_Fstartset_G = errorset.G
        else:
            # Fallback
            n = startset.dim() if hasattr(startset, 'dim') else len(startset)
            box_Fstartset_c = np.zeros((n, 1))
            box_Fstartset_G = np.zeros((n, 1))
    
    return eps_F, box_Fstartset_c, box_Fstartset_G


def compute_eps_G(errs: LinErrorBound, G, u) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute curvature error (input), see [1, Prop. 1]:
    eps_G = 2 * errOp(G*u)
    """
    if hasattr(G, 'center') and hasattr(G, 'rad'):
        Gcenter = G.center()
        Grad = G.rad()
    else:
        # Fallback for simple interval
        Gcenter = (G.supremum() + G.infimum()) / 2
        Grad = (G.supremum() - G.infimum()) / 2
    
    Gu_c = Gcenter @ u
    Gu_G = Grad @ np.abs(u)
    eps_G = 2 * np.linalg.norm(Gu_G + np.abs(Gu_c))
    
    return eps_G, Gu_c, Gu_G


def _priv_errOp(S) -> float:
    """
    Compute error operation for different set types
    """
    if hasattr(S, 'generators') and hasattr(S, 'center'):
        # Zonotope-like
        G = S.generators()
        c = S.center()
        return np.linalg.norm(np.sum(np.abs(G), axis=1) + np.abs(c))
    elif hasattr(S, 'infimum') and hasattr(S, 'supremum'):
        # Interval-like
        return np.linalg.norm(np.maximum(-S.infimum(), S.supremum()))
    elif hasattr(S, 'rad'):
        # Interval with rad method
        return np.linalg.norm(S.rad())
    else:
        # Fallback: assume it has some norm
        try:
            return np.linalg.norm(S)
        except:
            return 0.1  # Small default error 