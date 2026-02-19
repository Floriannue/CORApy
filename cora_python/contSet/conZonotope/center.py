"""
center - returns a point inside the constrained zonotope. The point is
   constructed from the chebychev-center of the polytope in the zonotope
   factor space

Syntax:
    c = center(cZ)
"""

import numpy as np
import scipy.linalg
from .conZonotope import ConZonotope


def center(cZ: ConZonotope) -> np.ndarray:
    """
    Returns a point inside the constrained zonotope.
    """
    # no constraints -> zonotope center
    if cZ.A.size == 0:
        return cZ.c

    # constraints -> compute chebychev center
    if cZ.representsa_('emptySet', np.finfo(float).eps):
        return np.zeros((0, 0))

    # check full-dimensionality (degenerate case handled separately)
    res_full_dim, _ = cZ.isFullDim()
    if not res_full_dim:
        # compute center of the 'cut' hypercube in factor space
        m = cZ.G.shape[1]
        Aineq = np.vstack([np.eye(m), -np.eye(m)])
        bineq = np.ones((2 * m, 1))
        Aeq = cZ.A
        beq = cZ.b

        from cora_python.contSet.polytope import Polytope
        P = Polytope(Aineq, bineq, Aeq, beq)
        c = P.center()
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        # map factor-space center back to state space
        return cZ.G @ c + cZ.c

    # construct inequality constraints for the unit cube
    n = cZ.G.shape[1]
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.ones((2 * n, 1))

    # calculate null space of the constraints
    Neq = scipy.linalg.null_space(cZ.A)

    # if null space is empty -> unique factors
    if Neq.size == 0:
        beta = np.linalg.lstsq(cZ.A, cZ.b, rcond=None)[0]
        return cZ.c + cZ.G @ beta

    # compute one point satisfying constraints
    x0 = np.linalg.pinv(cZ.A) @ cZ.b

    # transform constraints to the null space
    A_ = A @ Neq
    b_ = b - A @ x0

    from cora_python.contSet.polytope import Polytope
    P = Polytope(A_, b_)

    # compute chebychev center in null space
    c = P.center()
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    # convert center back to normal factor space
    c_ = Neq @ c + x0

    # compute center of the constrained zonotope in state space
    return cZ.c + cZ.G @ c_


