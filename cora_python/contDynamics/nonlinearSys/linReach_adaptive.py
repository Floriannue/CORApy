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
        minorder_val = options.get('minorder', 0)
        if not np.isfinite(minorder_val):
            minorder_idx = 0
        else:
            minorder_idx = int(minorder_val)
        zetaphi = options.get('zetaphi', [])
        if isinstance(zetaphi, (list, np.ndarray)) and len(zetaphi) > 0:
            minorder_idx = max(0, min(minorder_idx, len(zetaphi) - 1))
        finitehorizon = options['finitehorizon'][options['i'] - 2] * (
            1 + options['varphi'][options['i'] - 2] - options['zetaphi'][minorder_idx])
        finitehorizon = min(params['tFinal'] - options['t'], finitehorizon)
        if withinTol(finitehorizon, options['stepsize'][options['i'] - 2], 1e-3):
            finitehorizon = options['finitehorizon'][options['i'] - 2] * 1.1
        options['timeStep'] = finitehorizon
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

        while True:
            abscount += 1
            if np.any(error_adm):
                errG = np.diag(error_adm.flatten())
                Verror = Zonotope(np.zeros_like(error_adm), errG[:, np.any(errG, axis=0)])
                RallError, options = linsys.errorSolution_adaptive(options, Verror)
            else:
                RallError = Zonotope(np.zeros((nlnsys.nr_of_dims, 1)))

            try:
                if options['alg'] == 'lin':
                    Rmax = Rlinti + RallError
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

            perfIndCurr = np.max(trueError / error_adm) if np.any(error_adm) else 0
            if perfIndCurr <= 1 or not np.any(trueError):
                perfInds.append(perfIndCurr)
                options['Lconverged'] = True
                break
            elif perfIndCounter > 1:
                perfInds.append(perfIndCurr)
                if perfIndCounter > 2 and perfInds[-1] > perfInds[-2]:
                    options['Lconverged'] = False
                    break

            error_adm = 1.1 * trueError
            perfIndCounter += 1

        if not options['Lconverged']:
            options['timeStep'] = options['timeStep'] * 0.5
            finitehorizon = options['timeStep']
            error_adm = options['error_adm_horizon']
            continue

        if veryfirststep or not np.all(zeroWidthDim == options.get('zeroWidthDim', zeroWidthDim)):
            options['zeroWidthDim'] = zeroWidthDim
            options = _aux_getPowers(nlnsys, options, linsys, zeroWidthDim, VerrorDyn)

            Rerror, options = linsys.errorSolution_adaptive(options, VerrorDyn, VerrorStat)

        if isinstance(Rerror, PolyZonotope):
            abstrerr = np.sum(np.abs(Rerror.GI), axis=1)
        else:
            abstrerr = np.sum(np.abs(Rerror.generators()), axis=1)

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

                temp = abstrerr[options['orders'] == options['minorder']] / \
                    abstrerr_h[options['orders'] == options['minorder']]
                temp = temp[~np.isnan(temp)]
                varphi = np.max(temp) if temp.size > 0 else 0.0
                minorder_val = options.get('minorder', 0)
                if not np.isfinite(minorder_val):
                    minorder_idx = 0
                else:
                    minorder_idx = int(minorder_val)
                zetaphi = options.get('zetaphi', [])
                if isinstance(zetaphi, (list, np.ndarray)) and len(zetaphi) > 0:
                    minorder_idx = max(0, min(minorder_idx, len(zetaphi) - 1))
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
            if timeStepequalHorizon:
                temp = abstrerr[options['orders'] == options['minorder']] / \
                    abstrerr_h[options['orders'] == options['minorder']]
                options['varphi'][options['i'] - 1] = np.max(temp[~np.isnan(temp)])
            elif not lastStep:
                options['varphi'][options['i'] - 1] = _aux_varphiest(
                    finitehorizon, options['timeStep'], Rerror_h, Rerror,
                    options['decrFactor'], options['orders'], options['minorder']
                )
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
    kprimeest = bestIdxnew
    return float(deltatest), float(kprimeest)


def _aux_varphiest(horizon: float, deltat: float, Rerr_h: Any, Rerr_deltat: Any,
                   decrFactor: float, orders: np.ndarray, minorder: float) -> float:
    if isinstance(Rerr_h, Zonotope) and isinstance(Rerr_deltat, Zonotope):
        G_Rerr_h = Rerr_h.generators()
        G_Rerr_deltat = Rerr_deltat.generators()
        rerr1 = np.linalg.norm(np.sum(np.abs(G_Rerr_h[orders == minorder, :]), axis=1), 2)
        rerrk = np.linalg.norm(np.sum(np.abs(G_Rerr_deltat[orders == minorder, :]), axis=1), 2)
    else:
        rerr1 = np.linalg.norm(np.sum(np.abs(Rerr_h.GI), axis=1), 2)
        rerrk = np.linalg.norm(np.sum(np.abs(Rerr_deltat.GI), axis=1), 2)

    rhs = rerrk / rerr1
    varphi_lim = decrFactor ** (minorder + 1)

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
    options['orders'] = qi.flatten()
    options['minorder'] = np.min(options['orders'])
    return options


def _aux_deltaReach_adaptive(linsys: Any, Rinit: Any, options: Dict[str, Any]) -> Any:
    eAt = linsys.taylor.eAt
    F = linsys.taylor.F
    inputCorr = linsys.taylor.inputCorr
    Rtrans = linsys.taylor.Rtrans

    dim = len(F)
    Rhom_tp_delta = (eAt - np.eye(dim)) * Rinit + Rtrans

    O = PolyZonotope(np.zeros((dim, 1)), np.zeros((dim, 0)), [], np.zeros((0, 0)))
    Rdelta = O.enclose(Rhom_tp_delta) + F * Zonotope(Rinit) + inputCorr
    if options.get('isRV', False):
        Rdelta = Rdelta + linsys.taylor.RV
    return Rdelta
