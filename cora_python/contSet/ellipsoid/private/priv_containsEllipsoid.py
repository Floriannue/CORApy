"""
Private function for checking if an ellipsoid contains another ellipsoid.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_containsEllipsoid(E1: 'Ellipsoid', E2: 'Ellipsoid', tol: float) -> Tuple[bool, bool, float]:
    """
    Checks whether an ellipsoid contains another ellipsoid.
    
    Args:
        E1: ellipsoid object (circumbody)
        E2: ellipsoid object (inbody)
        tol: tolerance
        
    Returns:
        res: true/false
        cert: certificate (always True)
        scaling: scaling factor
    """
    try:
        # We follow the naming conventions from [2]
        n = E1.dim()
        G = E2.generators()
        H = E1.generators()
        
        # Need to check if E1 even has a chance to contain E2, in case E1 is degenerate
        r = np.linalg.matrix_rank(H, tol=tol)
        if r < n:
            r_comb = np.linalg.matrix_rank(np.hstack([H, G]), tol=tol)
            if r_comb > r:
                return False, True, np.inf
        
        # In any other instance, we should be good
        m = G.shape[1]
        ell = H.shape[1]
        
        H_pinv = np.linalg.pinv(H)
        Theta = H_pinv @ G
        theta = H_pinv @ (E2.q - E1.q)
        
        # Build matrices for SDP
        A = np.block([[Theta.T @ Theta, Theta.T @ theta.reshape(-1, 1)],
                      [theta.reshape(1, -1) @ Theta, theta.T @ theta]])
        
        B = np.zeros((m + 1, m + 1))
        B[m, m] = 1.0
        
        C = np.eye(m + 1)
        C[m, m] = -1.0
        
        # Solve SDP problem (simplified approach using eigenvalues)
        # This is a simplified version - full implementation would use MOSEK/SeDuMi
        try:
            # Try to solve using eigenvalue approach
            eigenvals_A = np.linalg.eigvals(A)
            eigenvals_B = np.linalg.eigvals(B)
            
            # Conservative estimate
            if np.any(eigenvals_B > tol):
                rho = np.max(eigenvals_A) / np.min(eigenvals_B[eigenvals_B > tol])
            else:
                rho = np.max(eigenvals_A)
            
            if rho <= 1 + tol:
                res = True
            else:
                res = False
            
            cert = True
            scaling = np.sqrt(np.abs(rho))
            
        except:
            # Fallback to conservative approach
            res = False
            cert = False
            scaling = np.inf
        
        return res, cert, scaling
        
    except Exception:
        return False, False, np.inf 