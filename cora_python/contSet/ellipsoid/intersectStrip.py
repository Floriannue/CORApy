"""
intersectStrip - computes the intersection between an ellipsoid and a 
   list of strips, where a strip is defined as | Cx - y | <= phi

Syntax:
   Eres = intersectStrip(E,C,phi,y[,sigma_sq_prev][,method])

Methods:
   'Gollamudi1996' and 'Liu2016'

Authors:       Matthias Althoff
Written:       06-January-2021
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def intersectStrip(E: Ellipsoid, C: np.ndarray, phi: Union[np.ndarray, object], y: np.ndarray, *varargin) -> Tuple[Ellipsoid, float]:
    # parse input
    (sigma_sq_prev, method), _ = setDefaultValues([0, 'Gollamudi1996'], list(varargin))
    if not isinstance(sigma_sq_prev, (int, float, np.floating)):
        method = sigma_sq_prev
        sigma_sq_prev = 0

    inputArgsCheck([
        [E, 'att', 'ellipsoid'],
        [C, 'att', 'numeric', 'matrix'],
        [phi, 'att', ['numeric', 'struct']],
        [y, 'att', 'numeric', 'column'],
        [sigma_sq_prev, 'att', 'numeric', 'scalar'],
        [method, 'str', ['Gollamudi1996', 'Liu2016']],
    ])

    # different methods for finding good lambda values
    if method == 'Gollamudi1996':
        if not (isinstance(phi, np.ndarray) and (phi.ndim == 1 or (phi.ndim == 2 and phi.shape[1] == 1))):
            raise CORAerror('CORA:wrongValue', 'third', 'numeric column')
        return _aux_methodGollamudi1996(E, C, phi.reshape(-1, 1), y.reshape(-1, 1), float(sigma_sq_prev))

    elif method == 'Liu2016':
        if not isinstance(phi, dict):
            raise CORAerror('CORA:wrongValue', 'third', 'struct')
        return _aux_methodLiu2016(E, C, phi, y.reshape(-1, 1), float(sigma_sq_prev))
    else:
        raise CORAerror('CORA:wrongValue', f"Unknown method '{method}'.")


def _aux_methodGollamudi1996(E: Ellipsoid, C: np.ndarray, phi: np.ndarray, y: np.ndarray, sigma_sq_prev: float) -> Tuple[Ellipsoid, float]:
    # obtain center and shape matrix
    c = E.center()
    Q = E.Q

    # auxiliary value
    delta = y - C @ c

    # identity matrix
    I = np.eye(y.shape[0])

    # Theorem 2 of [1]
    if sigma_sq_prev + float(delta.T @ delta) > float(phi):
        alpha = 1e-4
        # G = C Q C'
        G = C @ Q @ C.T
        # maximum singular value of G
        g = float(np.linalg.svd(G, compute_uv=False).max())

        # compute nu
        if withinTol(delta, 0):
            nu = alpha
        else:
            beta = (float(phi) - sigma_sq_prev) / float(delta.T @ delta)
            if withinTol(g, 1):
                nu = (1 - beta) / 2
            else:
                omega = 1 + beta * (g - 1)
                if omega > 0:
                    nu = 1 / (1 - g) * (1 - np.sqrt(g / omega))
                else:
                    nu = alpha
        # obtain lambda
        lam = min(alpha, nu)

        # new center
        c = c + lam * Q @ C.T @ delta

        # new sigma_sq
        Q_tmp = (1 - lam) * I + lam * G
        sigma_sq = 1 - lam + lam * float(phi) ** 2 - lam * (1 - lam) * float(delta.T @ np.linalg.inv(Q_tmp) @ delta)

        # new shape matrix
        invQ = (1 - lam) * np.linalg.pinv(Q) + lam * (C.T @ C)
        Q = np.linalg.pinv(invQ) / sigma_sq

        E.Q = Q
        E.q = c
    else:
        sigma_sq = sigma_sq_prev

    return E, sigma_sq


def _aux_methodLiu2016(E: Ellipsoid, C: np.ndarray, sys: dict, y: np.ndarray, sigma_sq_prev: float) -> Tuple[Ellipsoid, float]:
    # obtain center and shape matrix
    c = E.center()
    Q = E.Q

    # auxiliary value
    delta = y - C @ c

    # Theorem 3 of [2]
    if sigma_sq_prev + float(delta.T @ delta) > 1:
        # G
        G = sys['bar_V'] @ C @ Q @ C.T @ sys['bar_V'].T
        # max eigenvalue of G
        g = float(np.max(np.linalg.eigvalsh(G)))

        # lambda via Theorem 3
        beta = (1 - sigma_sq_prev) / float(delta.T @ delta)
        if withinTol(g, 1):
            lam = (1 - beta) / 2
        else:
            omega = 1 + beta * (g - 1)
            lam = 1 / (1 - g) * (1 - np.sqrt(g / omega))

        # Q_tmp (eq. 27)
        Q_tmp = (1 / lam) * sys['V'] + (1 / (1 - lam)) * (C @ Q @ C.T)
        Q_tmp_inv = np.linalg.pinv(Q_tmp)
        # K (eq. 26)
        K = (1 / (1 - lam)) * Q @ C.T @ Q_tmp_inv
        # sigma_sq (eq. 25)
        sigma_sq = (1 - lam) * sigma_sq_prev + lam - float(delta.T @ Q_tmp_inv @ delta)

        # new shape matrix (eq. 24)
        I = np.eye(Q.shape[0])
        Q = (1 - lam) * (I - K @ C) @ Q
        # new center (eq. 23)
        c = c + K @ delta

        E.Q = Q
        E.q = c
    else:
        sigma_sq = sigma_sq_prev

    return E, sigma_sq

