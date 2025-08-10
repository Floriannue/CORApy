"""
convHull_ - computes an overapproximation of the convex hull of an
   ellipsoid and another set representation

Syntax:
   E = convHull_(E,S)

Inputs:
   E - ellipsoid object
   S - contSet class object (or class array)

Outputs:
   S_out - convex hull

Authors:       Victor Gassmann
Written:       13-March-2019
Last update:   19-March-2021 (complete rewrite)
               04-July-2022 (input checks)
Last revision: 29-September-2024 (MW, integrate precedence)
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def convHull_(E: Ellipsoid, S, *varargin):
    # check inputs
    inputArgsCheck([
        [E, 'att', 'ellipsoid', ['scalar']],
        [S, 'att', ['contSet', 'numeric']],
    ])

    # ellipsoid is already convex
    if S is None:
        return E

    # ensure that numeric is second input argument
    E, S = reorder_numeric(E, S)

    # check dimensions
    equal_dim_check(E, S)

    # call function with lower precedence
    if hasattr(S, 'precedence') and S.precedence < E.precedence:
        # Defer to S's convHull where implemented
        if hasattr(S, 'convHull_'):
            return S.convHull_(E, *varargin)
        # If not implemented, fall through to errors below

    # convex hull with empty set
    if hasattr(S, 'representsa_') and S.representsa_('emptySet', 2.220446049250313e-16):
        return E

    # simply call 'or' for ellipsoid with ellipsoid
    if isinstance(S, Ellipsoid):
        return E.or_(S, 'outer')

    # no other cases implemented
    raise CORAerror('CORA:noops', E, S)

