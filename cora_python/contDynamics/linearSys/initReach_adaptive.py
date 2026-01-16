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

    Rend = {}

    # preliminary solutions without RV
    if 'gredIdx' in options and len(options['gredIdx'].get('Rhomti', [])) == options['i']:
        Rend['ti'] = Rhom.reduce('idx', options['gredIdx']['Rhomti'][options['i'] - 1])
    else:
        Rhom_res = Rhom.reduce('adaptive', options['redFactor'])
        if isinstance(Rhom_res, tuple):
            Rend['ti'], _, idx = Rhom_res
            if 'gredIdx' in options:
                options['gredIdx'].setdefault('Rhomti', []).append(idx)
        else:
            Rend['ti'] = Rhom_res

    if 'gredIdx' in options and len(options['gredIdx'].get('Rhomtp', [])) == options['i']:
        Rend['tp'] = Rhom_tp.reduce('idx', options['gredIdx']['Rhomtp'][options['i'] - 1])
    else:
        Rhomtp_res = Rhom_tp.reduce('adaptive', options['redFactor'])
        if isinstance(Rhomtp_res, tuple):
            Rend['tp'], _, idx = Rhomtp_res
            if 'gredIdx' in options:
                options['gredIdx'].setdefault('Rhomtp', []).append(idx)
        else:
            Rend['tp'] = Rhomtp_res

    # reduce and add RV only if exists
    if linOptions.get('isRV', False):
        if 'gredIdx' in options and len(options['gredIdx'].get('Rpar', [])) == options['i']:
            RV = linsys.taylor.RV.reduce('idx', options['gredIdx']['Rpar'][options['i'] - 1])
        else:
            RV_res = linsys.taylor.RV.reduce('adaptive', options['redFactor'])
            if isinstance(RV_res, tuple):
                RV, _, idx = RV_res
                if 'gredIdx' in options:
                    options['gredIdx'].setdefault('Rpar', []).append(idx)
            else:
                RV = RV_res

        Rend['ti'] = Rend['ti'] + RV
        Rend['tp'] = Rend['tp'] + RV

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
