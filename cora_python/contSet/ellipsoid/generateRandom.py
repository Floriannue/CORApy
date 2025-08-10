"""
generateRandom - Generates a random ellipsoid

Syntax:
   E = Ellipsoid.generateRandom()
   E = Ellipsoid.generateRandom('Dimension', n)
   E = Ellipsoid.generateRandom('Center', c)
   E = Ellipsoid.generateRandom('IsDegenerate', True)

Authors:       Victor Gassmann, Matthias Althoff
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np
from typing import Any

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.checkNameValuePairs import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing.readNameValuePair import readNameValuePair


def generateRandom(*args: Any) -> Ellipsoid:
    # name-value pairs -> number of input arguments is always even
    if len(args) % 2 != 0:
        raise CORAerror('CORA:evenNumberInputArgs')

    NVpairs = list(args)
    checkNameValuePairs(NVpairs, ['Dimension', 'IsDegenerate', 'Center'])
    NVpairs, n = readNameValuePair(NVpairs, 'Dimension')
    NVpairs, isdegenerate = readNameValuePair(NVpairs, 'IsDegenerate')
    NVpairs, q = readNameValuePair(NVpairs, 'Center')

    # defaults
    if n is None:
        if q is None:
            maxdim = 30
            n = np.random.randint(1, maxdim + 1)
        else:
            n = int(len(q))

    if q is None:
        q = np.random.randn(n, 1)
    else:
        q = np.asarray(q).reshape(-1, 1)

    if isdegenerate is None:
        isdegenerate = False

    # random symmetric PSD Q
    tmp = np.random.randn(n, n)
    Q = tmp.T @ tmp
    Q = 0.5 * (Q + Q.T)

    E0 = Ellipsoid.empty(1)
    TOL = E0.TOL

    if isdegenerate is not None:
        U, s, _ = np.linalg.svd(Q)
        sv_min = np.sqrt(TOL) * np.max(s)
        max_sv = np.max(s)
        ind_ts = s < sv_min
        s[ind_ts] = max_sv + 0.5 * (max_sv - sv_min) * np.random.rand(np.sum(ind_ts))
        if isdegenerate:
            n_d = np.random.randint(1, n + 1)
            dims = np.random.randint(0, n, size=n_d)
            s[dims] = 0
        Q = U @ np.diag(s) @ U.T

    Q = 0.5 * (Q + Q.T)

    maxAbsValue = 10.0
    if np.any(Q > maxAbsValue) or np.any(Q < -maxAbsValue):
        factor = 10.0 / max(np.max(np.abs(Q)), 1e-12)
        Q = Q * factor

    return Ellipsoid(Q, q)

