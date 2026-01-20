"""
reach_adaptive - computes the reachable continuous set

Syntax:
    [timeInt,timePoint,res,tVec,options] = reach_adaptive(nlnsys,params,options)

Inputs:
    nlnsys - nonlinearSys object
    params - model parameters
    options - options for the computation of reachable sets

Outputs:
    timeInt - cell-array of time-interval solutions
    timePoint - cell-array of time-point solutions
    res - satisfaction / violation of specifications
    tVec - vector of time steps
    options - options for the computation of reachable sets (param tracking)

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       27-May-2020 (MATLAB)
Last update:   25-February-2021 (merge to master)
Python translation: 2025
"""

from typing import Any, Dict, Tuple, List
import numpy as np

from cora_python.contSet.interval import Interval
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.verbose.verboseLog import verboseLog


def reach_adaptive(nlnsys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, List], bool, np.ndarray, Dict[str, Any]]:
    # initialize cell-arrays that store the reachable set
    timeInt = {'set': [], 'time': []}
    timePoint = {'set': [params['R0']], 'time': [params['tStart']]}
    res = False
    tVec = []

    # remove 'adaptive' from alg (just for tensor computation)
    if 'lin' in options['alg']:
        options['alg'] = 'lin'
    elif 'poly' in options['alg']:
        options['alg'] = 'poly'

    # iteration counter and time for main loop
    options['i'] = 1
    options['t'] = params['tStart']
    options['R'] = params['R0']
    options['error_adm_horizon'] = np.zeros((nlnsys.nr_of_dims, 1))
    options['error_adm_Deltatopt'] = np.zeros((nlnsys.nr_of_dims, 1))
    abortAnalysis = False

    # MAIN LOOP
    while params['tFinal'] - options['t'] > 1e-12 and not abortAnalysis:
        verboseLog(options.get('verbose', 0), options['i'], options['t'], params['tStart'], params['tFinal'])

        # reduction of R via restructuring (only poly)
        if isinstance(options['R'], PolyZonotope):
            ratio = options['R'].approxVolumeRatio(options['polyZono']['volApproxMethod'])
            if ratio > options['polyZono']['maxPolyZonoRatio']:
                options['R'] = options['R'].restructure(
                    options['polyZono']['restructureTechnique'],
                    options['polyZono']['maxDepGenOrder']
                )

        # propagation of reachable set
        if options.get('progress', False):
            print(f"[reach_adaptive] Starting linReach_adaptive for step {options['i']}...", flush=True)
        Rti, Rtp, options = nlnsys.linReach_adaptive(options['R'], params, options)
        if options.get('progress', False):
            print(f"[reach_adaptive] Completed linReach_adaptive for step {options['i']}", flush=True)
        Rnext = {'ti': Rti, 'tp': Rtp}

        # reduction for next step
        Rti_res = Rnext['ti'].reduce('adaptive', options['redFactor'] * 5)
        Rnext['ti'] = Rti_res[0] if isinstance(Rti_res, tuple) else Rti_res
        Rtp_res = Rnext['tp'].reduce('adaptive', options['redFactor'])
        Rnext['tp'] = Rtp_res[0] if isinstance(Rtp_res, tuple) else Rtp_res

        # additional reduction for poly
        if isinstance(Rnext['tp'], PolyZonotope):
            Rnext['tp'] = _aux_reduceOnlyDep(Rnext['tp'], options['polyZono']['maxDepGenOrder'])

        # optional progress logging (helps diagnose non-termination)
        if options.get('progress', False):
            interval = int(options.get('progressInterval', 10))
            if options['i'] == 1 or (interval > 0 and options['i'] % interval == 0):
                time_step = options.get('timeStep', np.nan)
                print(f"[reach_adaptive] step={options['i']} t={options['t']:.6g} dt={time_step:.6g}")

        # save to output variables
        tVec.append(options['timeStep'])
        timeInt['set'].append(Rnext['ti'])
        timeInt['time'].append(Interval(options['t'], options['t'] + tVec[-1]))
        timePoint['set'].append(Rnext['tp'])
        timePoint['time'].append(options['t'] + tVec[-1])

        # increment time
        options['t'] = options['t'] + options['timeStep']
        options['i'] = options['i'] + 1

        # start set for next step (since always initReach called)
        options['R'] = Rnext['tp']
        
        # Debug: Track reachable set size to identify when it starts growing unbounded
        if options.get('progress', False) and (options['i'] % 10 == 0 or options['i'] <= 5):
            try:
                R_center = options['R'].center()
                R_radius = np.linalg.norm(options['R'].interval().rad())
                max_abs_center = np.max(np.abs(R_center))
                print(f"[reach_adaptive] Step {options['i']}: R center max abs = {max_abs_center:.6e}, R radius = {R_radius:.6e}", flush=True)
                if max_abs_center > 1e+50:
                    print(f"[reach_adaptive] WARNING: Reachable set center is extremely large! This may indicate numerical instability.", flush=True)
            except Exception as e:
                print(f"[reach_adaptive] Could not compute R size: {e}", flush=True)

        # check for timeStep -> 0
        abortAnalysis = _aux_checkForAbortion(np.array(tVec), options['t'], params['tFinal'])

    verboseLog(options.get('verbose', 0), options['i'], options['t'], params['tStart'], params['tFinal'])
    return timeInt, timePoint, res, np.array(tVec), options


def _aux_checkForAbortion(tVec: np.ndarray, currt: float, tFinal: float) -> bool:
    abortAnalysis = False
    remTime = tFinal - currt
    N = 10
    k = len(tVec)
    if k == 0:
        return False
    lastNsteps = np.sum(tVec[max(0, k - N):])
    if lastNsteps == 0:
        return True
    if remTime / lastNsteps > 1e9:
        abortAnalysis = True
    return abortAnalysis


def _aux_reduceOnlyDep(R: PolyZonotope, order: int) -> PolyZonotope:
    n, Gsize = R.G.shape
    if Gsize / n < order + 1:
        return R

    h = np.linalg.norm(R.G, ord=2, axis=0)
    ind = np.argsort(h)[::-1]
    ind = ind[order * n:]

    Gred = R.G[:, ind]
    Ered = R.E[:, ind]
    pZred = PolyZonotope(np.zeros((n, 1)), Gred, [], Ered)

    zono = Zonotope(pZred)
    zono = Zonotope(zono.c, np.diag(np.sum(np.abs(zono.G), axis=1)))

    Grem = R.G.copy()
    Grem = np.delete(Grem, ind, axis=1)
    Erem = R.E.copy()
    Erem = np.delete(Erem, ind, axis=1)

    newc = R.c + zono.c
    GInew = np.hstack([R.GI, zono.G]) if R.GI.size > 0 else zono.G
    return PolyZonotope(newc, Grem, GInew, Erem)
