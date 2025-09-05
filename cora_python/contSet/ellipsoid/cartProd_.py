"""
cartProd_ - computes the Cartesian product of an ellipsoid and another set or point

Syntax:
   E = cartProd_(E,S)
   E = cartProd_(E,S,mode)

Inputs:
   E - ellipsoid object
   S - contSet object, numeric
   mode - type of approximation: 'exact', 'outer', 'inner'

Outputs:
   E - ellipsoid object

Example:
   E1 = ellipsoid([3 -1; -1 1],[1;0]);
   E2 = ellipsoid([5 1; 1 2],[1;-1]);
   E = cartProd(E1,E2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/cartProd

Authors:       Victor Gassmann
Written:       19-March-2021
Last update:   02-June-2022 (VG, handle empty case)
               17-April-2024 (TL, simplified ellipsoid-ellipsoid case)
Last revision: 27-March-2023 (MW, rename cartProd_)
               22-September-2024 (MW, restructure)
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric


def cartProd_(E, S, mode: Optional[str] = None):
    """
    Python translation of MATLAB ellipsoid/cartProd_.m
    """

    # default mode
    mode_val = setDefaultValues(['outer'], mode)[0]
    mode = mode_val

    # Only 'outer' is supported for Ellipsoid per MATLAB
    if mode in ('inner', 'exact'):
        raise CORAerror(
            'CORA:notSupported',
            "The function 'cartProd' supports only type = 'outer' for ellipsoid objects."
        )

    if mode != 'outer':
        # any other mode -> not supported
        raise CORAerror(
            'CORA:notSupported',
            "The function 'cartProd' supports only type = 'outer' for ellipsoid objects."
        )

    # Ensure correct ordering for numeric
    E, S = reorder_numeric(E, S)

    # Input checks (mimic MATLAB semantics where applicable)
    # E must be ellipsoid or numeric column for the numeric-ellipsoid branch
    # S can be ellipsoid or numeric column
    # We validate dimensions when needed, otherwise let specific branches handle
    # equalDimCheck only when both are contSets
    # Dimension checks are handled in the specific branches as needed

    # ellipsoid-ellipsoid case: Cartesian product of interval over-approximations
    if isinstance(E, Ellipsoid) and isinstance(S, Ellipsoid):
        # Construct intervals via ellipsoid -> interval conversion and
        # use exact cartProd_ there, then convert back to ellipsoid
        IE = E.interval()
        IS = S.interval()
        I_out = IE.cartProd_(IS, 'exact')
        return I_out.ellipsoid()

    # ellipsoid-numeric case: numeric must be column vector
    if isinstance(E, Ellipsoid) and isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == 1:
        # block-diagonal Q with zeros for numeric part; concatenated centers
        n_extra = S.shape[0]
        Q_out = np.block([
            [E.Q, np.zeros((E.Q.shape[0], n_extra))],
            [np.zeros((n_extra, E.Q.shape[0])), np.zeros((n_extra, n_extra))],
        ])
        q_out = np.vstack([E.q, S.reshape(-1, 1)])
        return Ellipsoid(Q_out, q_out)

    # numeric-ellipsoid case
    if isinstance(E, np.ndarray) and E.ndim == 2 and E.shape[1] == 1 and isinstance(S, Ellipsoid):
        n_extra = E.shape[0]
        Q_out = np.block([
            [np.zeros((n_extra, n_extra)), np.zeros((n_extra, S.Q.shape[0]))],
            [np.zeros((S.Q.shape[0], n_extra)), S.Q],
        ])
        q_out = np.vstack([E.reshape(-1, 1), S.q])
        return Ellipsoid(Q_out, q_out)

    # All other cases not implemented
    raise CORAerror('CORA:noops', E, S, mode)

