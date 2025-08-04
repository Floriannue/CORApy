"""
This module contains the function for computing the ellipsoid norm.
"""

import numpy as np
from scipy.linalg import svd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def ellipsoidNorm(E: 'Ellipsoid', p: np.ndarray) -> float:
    """
    ellipsoidNorm - computes the norm of the point p w.r.t. the
    ellipsoid-norm induced by the ellipsoid E; this is defined similarly
    to the zonotope-norm defined in [1, Definition 4].

    Syntax:
        res = ellipsoidNorm(E,p)

    Inputs:
        E - ellipsoid
        p - nx1-array, with n the dimension of E

    Outputs:
        res - ellipsoid-norm of the point p
    """

    if E.isemptyobject():
        return np.inf

    if p.shape[0] != E.dim():
        raise ValueError("Dimension of point p must match dimension of ellipsoid E.")
    
    # Ensure p is a column vector
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    # The ellipsoid E is given as {x | (x - q)' * Q^(-1) (x - q) <= 1}...
    # ... as a consequence, the norm of a point p w.r.t. E would be given by
    # p'*Q^(-1)*p, since we have to ignore the center to get a norm.
    # So the only thing we need to be careful about is when E is degenerate,
    # then we may need to invert Q by hand.
    # In theory, we would then need to compute
    # res = sqrt(abs(p'*Qinv*p))
    # However, this is not quite stable for most cases. That is why we compute
    # instead
    # q = Q\p;
    # res = sqrt(abs(p'*q))
    # for the case where Q is invertible; otherwise, we do something similar
    # for the degenerate case

    if not E.isFullDim():
        # Degenerate case: use SVD
        U, S, Vt = np.linalg.svd(E.Q)
        s = S  # S is already a 1D array of singular values
        s[s < 1e-10] = 0
        
        # Create diagonal matrix for inverse
        n = len(s)
        Sinv = np.zeros((n, n))
        for i in range(n):
            if s[i] > 1e-10:
                Sinv[i, i] = 1.0 / s[i]
        
        # We consider the ellipsoid E that has been turned by U', such that
        # the axes of E coincide with the canonical ONB; therefore, we also
        # need to rotate p:
        p_rot = U.T @ p
        # For numerical reasons, we need to manually set coordinates of p that
        # are very small to zero
        p_rot[np.abs(p_rot) < 1e-10] = 0
        
        q = Sinv @ p_rot
        # Now, we need to be careful; Qinv will contain Inf, and p may contain
        # 0, which is the only case that can make the point contained. Thus,
        # we need to set the values of q to zero in this case
        q[np.isnan(q)] = 0
        
        # Debug: Check if any component of p_rot is non-zero in a direction where s is zero
        # This would indicate the point is outside the degenerate subspace
        for i in range(n):
            if s[i] == 0 and abs(p_rot[i]) > 1e-10:
                return np.inf
    else:
        # Non-degenerate case
        q = np.linalg.solve(E.Q, p)

    res = np.sqrt(np.abs(p.T @ q))
    # The square root is just there to make sure that the resulting function is
    # a norm (i.e., scaling p by a factor a should yield a|p|, not a^2|p|.

    # We need to do one last check: If res = NaN, something weird happened with
    # additions and subtractions of Inf. However, this means that some Inf was
    # left, so the point has no chance of being contained. Thus, set it
    # manually:
    if np.isnan(res):
        res = np.inf

    return float(res.item() if hasattr(res, 'item') else res) 