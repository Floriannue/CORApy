"""
zonoBundle - convert a polytope to a zonotope bundle (MATLAB parity)

Throws if fullspace/unbounded. For bounded polytopes, constructs a bundle by
enclosing the vertex set under each constraint normal's orthogonal basis.
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def zonoBundle(P: 'Polytope'):
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope.zonotope import Zonotope
    from cora_python.contSet.interval.interval import Interval

    # Fullspace / unbounded check
    if P.representsa_('fullspace', 0) or not P.isBounded():
        raise CORAerror('CORA:specialError', 'Polytope is unbounded and cannot be converted into a zonotope bundle.')

    # Ensure constraints and vertices
    P.constraints()
    V = P.vertices_()
    if V.size == 0:
        return ZonoBundle([])

    # Build bundle per constraint rows (A; Ae; -Ae)
    A = np.vstack([P.A, P.Ae, -P.Ae]) if (P.A.size + P.Ae.size) > 0 else np.zeros((0, P.dim()))
    Z_list = []
    for i in range(A.shape[0]):
        a = A[i, :].reshape(-1, 1)
        if np.linalg.norm(a) < 1e-14:
            continue
        # Orthonormal basis with first vector along a
        # Use Gram-Schmidt via SVD for stability
        u = a / np.linalg.norm(a)
        # Complete to orthonormal basis using QR on a random matrix augmented with u
        n = P.dim()
        M = np.eye(n)
        M[:, 0:1] = u
        Q, _ = np.linalg.qr(M)
        B = Q  # n x n basis
        V_t = B.T @ V
        lb = np.min(V_t, axis=1, keepdims=True)
        ub = np.max(V_t, axis=1, keepdims=True)
        I = Interval(lb, ub)
        Z = B @ Zonotope(I)
        Z_list.append(Z)

    return ZonoBundle(Z_list)


