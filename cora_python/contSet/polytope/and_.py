"""
and_ - Intersection of two polytopes (P1 & P2) (MATLAB parity)

Constructs the intersection in H-rep by stacking constraints.
If either is empty, returns empty. If both unconstrained fullspaces, returns fullspace.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope


def and_(P1: 'Polytope', P2: 'Polytope') -> 'Polytope':
    from .polytope import Polytope

    # Ensure constraints
    P1.constraints(); P2.constraints()

    n = max(P1.dim(), P2.dim())

    # Stack constraints: A = [A1; A2], Ae = [Ae1; Ae2]
    def vstack_or_empty(X, Y, cols):
        if X.size == 0 and Y.size == 0:
            return np.zeros((0, cols))
        if X.size == 0:
            return Y
        if Y.size == 0:
            return X
        return np.vstack([X, Y])

    A = vstack_or_empty(P1.A, P2.A, n)
    b = vstack_or_empty(P1.b.reshape(-1, 1), P2.b.reshape(-1, 1), 1)
    Ae = vstack_or_empty(P1.Ae, P2.Ae, n)
    be = vstack_or_empty(P1.be.reshape(-1, 1), P2.be.reshape(-1, 1), 1)

    return Polytope(A, b, Ae, be)


