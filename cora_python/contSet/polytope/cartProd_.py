"""
cartProd_ - Cartesian product of two polytopes (MATLAB parity)

Syntax:
    P = cartProd_(P1, P2)

For P1 in R^n and P2 in R^m, returns P in R^{n+m} such that
    P = {(x,y) | A1 x <= b1, Ae1 x = be1, A2 y <= b2, Ae2 y = be2}

Authors: MATLAB CORA team; Python translation by AI Assistant
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def cartProd_(P1: 'Polytope', P2: 'Polytope') -> 'Polytope':
    from .polytope import Polytope

    n1 = P1.dim()
    n2 = P2.dim()

    # Ensure constraints available
    P1.constraints(); P2.constraints()

    # Inequalities
    A1, b1 = P1.A, P1.b
    A2, b2 = P2.A, P2.b
    # Equalities
    Ae1, be1 = P1.Ae, P1.be
    Ae2, be2 = P2.Ae, P2.be

    # Build block-diagonal constraints
    def blk(A, B):
        if A.size == 0 and B.size == 0:
            return np.zeros((0, n1 + n2))
        if A.size == 0:
            return np.hstack([np.zeros((B.shape[0], n1)), B])
        if B.size == 0:
            return np.hstack([A, np.zeros((A.shape[0], n2))])
        top = np.hstack([A, np.zeros((A.shape[0], n2))])
        bot = np.hstack([np.zeros((B.shape[0], n1)), B])
        return np.vstack([top, bot])

    A = blk(A1, A2)
    b = np.vstack([b1.reshape(-1, 1) if b1.size > 0 else np.zeros((0, 1)),
                   b2.reshape(-1, 1) if b2.size > 0 else np.zeros((0, 1))])

    Ae = blk(Ae1, Ae2)
    be = np.vstack([be1.reshape(-1, 1) if be1.size > 0 else np.zeros((0, 1)),
                    be2.reshape(-1, 1) if be2.size > 0 else np.zeros((0, 1))])

    return Polytope(A, b, Ae, be)


