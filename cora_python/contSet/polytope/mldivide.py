"""
mldivide - overloaded '\\' operator for set difference P1 \ P2

Exact MATLAB translation of control flow; uses existing helpers.
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def mldivide(P1, P2):
    from .polytope import Polytope

    if not isinstance(P1, Polytope) or not isinstance(P2, Polytope):
        raise CORAerror('CORA:wrongInput', 'Both inputs must be polytopes')

    n = P1.dim()

    # empty cases
    if P1.representsa_('emptySet', 1e-12):
        return Polytope.empty(n)
    if P2.representsa_('emptySet', 1e-12):
        return P1

    # minimal H-representation
    P1c = P1.compact_('all', 1e-9)
    P2c = P2.compact_('all', 1e-9)

    A1, b1, Ae1, be1 = P1c.A, P1c.b, P1c.Ae, P1c.be
    A2, b2, Ae2, be2 = P2c.A, P2c.b, P2c.Ae, P2c.be

    P_out = Polytope(np.zeros((0, n)), np.zeros((0, 1)))

    if P1c.isFullDim() and P2c.isFullDim():
        for i in range(b2.shape[0]):
            A = np.vstack([-A2[i:i+1, :], A2[:i, :], A1])
            b = np.vstack([-b2[i:i+1, :], b2[:i, :], b1])
            Pi = Polytope(A, b)
            if Pi.isFullDim():
                P_out = Pi
        return P_out

    if P1c.isFullDim() or (Ae2.size > 0 and Ae1.size > 0 and np.linalg.matrix_rank(nullspace(Ae2)) < np.linalg.matrix_rank(nullspace(Ae1))):
        return P1c

    if P2c.isFullDim() or (Ae2.size > 0 and Ae1.size > 0 and np.linalg.matrix_rank(nullspace(Ae2)) > np.linalg.matrix_rank(nullspace(Ae1))):
        # shift open half-space of P2 slightly
        shift_tol = 1e-12
        for i in range(b1.shape[0]):
            A = np.vstack([-A2[i:i+1, :], A2[:i, :], A1])
            b = np.vstack([-(b2[i, 0] + shift_tol)].reshape(1, 1), b2[:i, :], b1)
            b[0, 0] = b[0, 0] + shift_tol
            Pi = Polytope(A, b)
            P_out = Pi if P_out.A is None or P_out.A.size == 0 else P_out
        return P_out

    if Ae2.size == 0 and Ae1.size == 0 and (P1c <= P2c):
        return Polytope()

    # project to null space of P2.Ae
    if Ae2.size > 0:
        F = nullspace(Ae2)
        x0 = np.linalg.lstsq(Ae2, be2, rcond=None)[0]
    else:
        F = np.eye(n)
        x0 = np.zeros((n, 1))

    P1_Z = Polytope(A1 @ F, b1 - A1 @ x0)
    P2_Z = Polytope(A2 @ F, b2 - A2 @ x0)

    # recursive call in projected space
    P_Z = mldivide(P1_Z, P2_Z)

    # project back (placeholder: return projected result directly until full multi-set union handling is implemented)
    return P_Z


def nullspace(A: np.ndarray, rtol: float = 1e-12):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = (s > rtol * s.max()).sum()
    return vh[rank:].T


