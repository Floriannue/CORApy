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

    # Special handling for zero-rank ellipsoids (points)
    # If it's a point ellipsoid (Q is zero matrix), norm is always infinity unless it's exactly the center
    # MATLAB's ellipsoidNorm for point ellipsoid returns Inf even for the center itself
    if E.representsa_('point', np.finfo(float).eps): 
        return np.inf

    if not E.isFullDim():
        U, s_vals, Vh = np.linalg.svd(E.Q)
        # Take the square root of singular values and handle potential negative values from numerical issues
        s_sqrt = np.sqrt(np.maximum(0, s_vals))

        # Calculate the norm: p' * U * inv(diag(s_vals)) * U' * p
        p_rot = U.T @ p
        print(f"ellipsoidNorm: p_shifted\n{p}\np_rot\n{p_rot}") # Debug print

        s_inv = np.zeros_like(s_vals)
        mask = s_vals > 1e-10 # MATLAB tolerance
        s_inv[mask] = 1.0 / s_vals[mask]
        print(f"ellipsoidNorm: s_vals\n{s_vals}\ns_inv\n{s_inv}") # Debug print
        
        res_sq = np.sum(s_inv * (p_rot**2))
        res = np.sqrt(res_sq)
        print(f"ellipsoidNorm: res_sq {res_sq}, res {res}") # Debug print
        
    else:
        # Non-degenerate case
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