"""
isIntersecting_ - determines if an ellipsoid intersects a set

Syntax:
   res = isIntersecting_(E,S,type,tol)

Inputs:
   E - ellipsoid object
   S - contSet object or double matrix (ncols = number of points)
   type - type of check ('exact' or 'approx')
   tol - tolerance

Outputs:
   res - true/false

Authors:       Victor Gassmann, Niklas Kochdumper
Written:       13-March-2019
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def isIntersecting_(E: Ellipsoid, S, type: str = 'exact', tol: float = 1e-9, *varargin):
    # ensure that numeric is second input argument
    E, S = reorder_numeric(E, S)

    # call function with lower precedence
    if hasattr(S, 'precedence') and S.precedence < E.precedence:
        return S.isIntersecting_(E, type, tol)

    # numeric case: check containment
    if isinstance(S, (list, tuple)) or (hasattr(S, 'ndim') and isinstance(S, np.ndarray)):
        return E.contains_(S, type, tol, 0, False, False)

    # sets must not be empty
    if E.representsa_('emptySet', 0) or (hasattr(S, 'representsa_') and S.representsa_('emptySet', 0)):
        return False

    # ellipsoid is just a point, check containment
    if E.representsa_('point', tol):
        return S.contains_(E.q, type, tol, 0, False, False)

    # general method: use shortest distance (must be 0 for intersection)
    try:
        dist = E.distance(S)
        return (dist < E.TOL) or (abs(dist - E.TOL) <= E.TOL)
    except Exception:
        if type == 'exact':
            raise CORAerror('CORA:noExactAlg', E, S)

        # use fallback if available
        if hasattr(S, 'quadMap'):
            from cora_python.contSet.ellipsoid.private.priv_isIntersectingMixed import priv_isIntersectingMixed
            return priv_isIntersectingMixed(E, S)
        raise CORAerror('CORA:noops', E, S)

