"""
polygon - convert a set to a (inner-approximative) polygon

Syntax:
   pgon = polygon(E)

Authors:       Tobias Ladner
Written:       14-October-2024
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def polygon(E: Ellipsoid, *varargin):
    # check dimension of set
    if E.dim() != 2:
        raise CORAerror('CORA:wrongValue', 'first', 'Given set must be 2-dimensional.')

    # compute boundary points
    if E.representsa_('point', np.finfo(float).eps):
        V = E.q
    else:
        N = 1000
        # compute boundary points via sampling directions on the unit circle
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        V = np.zeros((2, N))
        Q = E.Q
        q = E.q.flatten()
        # For each angle, direction d, boundary point x = q + t*Q*d with t s.t. (x-q)'Q^{-1}(x-q)=1
        # Closed-form: boundary point is q + (Q @ d) / sqrt(d' Q d)
        for k, ang in enumerate(angles):
            d = np.array([np.cos(ang), np.sin(ang)])
            denom = np.sqrt(d.T @ Q @ d)
            if denom <= 0:
                V[:, k] = q
            else:
                V[:, k] = q + (Q @ d) / denom

    # init polygon
    from cora_python.contSet.polygon import Polygon
    return Polygon(V)

