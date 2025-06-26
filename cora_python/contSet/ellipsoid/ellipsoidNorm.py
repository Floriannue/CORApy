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

    if not E.isFullDim():
        U, s_vals, _ = svd(E.Q)
        
        # We need to compute p' * pinv(Q) * p
        # pinv(Q) = U * diag(1/s_i if s_i > tol else 0) * U'
        # So p' * pinv(Q) * p = (U'p)' * diag(1/s_i) * (U'p)
        
        p_rot = U.T @ p
        
        s_inv = np.zeros_like(s_vals)
        mask = s_vals > 1e-10 # MATLAB tolerance
        s_inv[mask] = 1.0 / s_vals[mask]
        
        # If a component of p_rot is non-zero where s_inv is zero (degenerate dir),
        # the point is outside the subspace spanned by the ellipsoid. The norm is Inf.
        if np.any(np.abs(p_rot[~mask]) > 1e-10):
            return np.inf

        # Equivalent to p_rot.T @ np.diag(s_inv) @ p_rot
        res_sq = np.sum(s_inv * (p_rot**2))
        
    else:
        try:
            q_vec = np.linalg.solve(E.Q, p)
            res_sq = p.T @ q_vec
        except np.linalg.LinAlgError:
            # Fallback for singular matrix that escaped is_full_dim check
            # This logic branch should ideally not be hit
            q_vec = np.linalg.pinv(E.Q) @ p
            res_sq = p.T @ q_vec

    res = np.sqrt(np.abs(res_sq))

    if np.isnan(res):
        res = np.inf

    return float(res) 