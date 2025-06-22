"""
This module contains the function for computing the ellipsoid norm.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def ellipsoidNorm(E: 'Ellipsoid', p: np.ndarray) -> float:
    """
    Computes the norm of the point p w.r.t. the ellipsoid-norm induced by the ellipsoid E.
    
    This is defined similarly to the zonotope-norm defined in [1, Definition 4].
    
    Args:
        E: ellipsoid
        p: point (n x 1 array), with n the dimension of E
        
    Returns:
        res: ellipsoid-norm of the point p
        
    References:
        [1] A. Kulmburg, M. Althoff. "On the co-NP-Completeness of the
            Zonotope Containment Problem", European Journal of Control 2021
    """
    # Input validation
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    
    if p.shape[0] != E.dim():
        raise ValueError(f"Point dimension {p.shape[0]} does not match ellipsoid dimension {E.dim()}")
    
    # Handle empty ellipsoids - they have infinite norm for any point
    if E.isemptyobject():
        return np.inf
    
    # Translate point relative to ellipsoid center
    # The ellipsoid norm should be computed for (p - q), not p
    p_centered = p - E.q
    
    # The ellipsoid E is given as {x | (x - q)' * Q^(-1) (x - q) <= 1}...
    # ... as a consequence, the norm of a point p w.r.t. E would be given by
    # (p-q)'*Q^(-1)*(p-q), since we need to consider the center offset.
    # So the only thing we need to be careful about is when E is degenerate,
    # then we may need to invert Q by hand.
    # In theory, we would then need to compute
    # res = sqrt(abs((p-q)'*Qinv*(p-q)))
    # However, this is not quite stable for most cases. That is why we compute
    # instead
    # q = Q\(p-q);
    # res = sqrt(abs((p-q)'*q))
    # for the case where Q is invertible; otherwise, we do something similar
    # for the degenerate case
    
    if not E.isFullDim():
        # Handle degenerate case using SVD
        U, S, Vt = np.linalg.svd(E.Q)
        
        # S is a 1D array of singular values, create proper inverse
        # For degenerate directions (S <= 1e-10), we want Inf so that
        # any non-zero component in that direction gives Inf norm
        s_inv = np.zeros_like(S)
        nonzero_mask = S > 1e-10
        s_inv[nonzero_mask] = 1.0 / S[nonzero_mask]
        s_inv[~nonzero_mask] = np.inf  # Infinite for degenerate directions
        
        # We consider the ellipsoid E that has been turned by U', such that
        # the axes of E coincide with the canonical ONB; therefore, we also
        # need to rotate (p-q):
        p_rot = U.T @ p_centered
        # For numerical reasons, we need to manually set coordinates of p that
        # are very small to zero
        p_rot[np.abs(p_rot) < 1e-10] = 0
        
        # Apply the inverse singular values
        # This will give Inf if p_rot has non-zero components in degenerate directions
        q_rot = s_inv.reshape(-1, 1) * p_rot
        
        # Now, we need to be careful; s_inv contains Inf, and p_rot may contain
        # 0, which is the only case that can make the point contained. Thus,
        # we need to set the values of q_rot to zero in this case (Inf * 0 = NaN -> 0)
        q_rot[np.isnan(q_rot)] = 0
        
        # Rotate back to original coordinate system
        q = U @ q_rot
        
        res = np.sqrt(np.abs(p_centered.T @ q).item())
    else:
        # Full-dimensional case
        try:
            q = np.linalg.solve(E.Q, p_centered)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            q = np.linalg.pinv(E.Q) @ p_centered
        
        res = np.sqrt(np.abs(p_centered.T @ q).item())
    
    # The square root is just there to make sure that the resulting function is
    # a norm (i.e., scaling p by a factor a should yield a|p|, not a^2|p|.
    
    # We need to do one last check: If res = NaN, something weird happened with
    # additions and subtractions of Inf. However, this means that some Inf was
    # left, so the point has no chance of being contained. Thus, set it
    # manually:
    if np.isnan(res):
        res = np.inf
    
    return res 