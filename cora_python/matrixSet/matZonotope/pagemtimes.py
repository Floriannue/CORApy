"""
pagemtimes - page-wise matrix multiplication (helper function)

This is a helper function to replicate MATLAB's pagemtimes functionality
for page-wise matrix multiplication of 3D arrays.

Authors: Python translation: 2025
"""

import numpy as np


def pagemtimes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Page-wise matrix multiplication
    
    For 3D arrays A (n x m x p) and B (m x k x p), computes:
    C[:, :, i] = A[:, :, i] @ B[:, :, i] for i = 0..p-1
    
    Args:
        A: First matrix array (can be 2D or 3D)
        B: Second matrix array (can be 2D or 3D)
        
    Returns:
        C: Result of page-wise multiplication
    """
    # Handle 2D case
    if A.ndim == 2 and B.ndim == 2:
        return A @ B
    
    # Handle 3D case
    if A.ndim == 3 and B.ndim == 3:
        n, m, p = A.shape
        m2, k, p2 = B.shape
        if m != m2 or p != p2:
            raise ValueError(f"Matrix dimensions must match: A.shape={A.shape}, B.shape={B.shape}")
        
        # Compute page-wise: C[:, :, i] = A[:, :, i] @ B[:, :, i]
        C = np.zeros((n, k, p))
        for i in range(p):
            C[:, :, i] = A[:, :, i] @ B[:, :, i]
        return C
    
    # Handle mixed 2D/3D cases
    if A.ndim == 2 and B.ndim == 3:
        # A is 2D, B is 3D: broadcast A to all pages
        n, m = A.shape
        m2, k, p = B.shape
        if m != m2:
            raise ValueError(f"Matrix dimensions must match: A.shape={A.shape}, B.shape={B.shape}")
        C = np.zeros((n, k, p))
        for i in range(p):
            C[:, :, i] = A @ B[:, :, i]
        return C
    
    if A.ndim == 3 and B.ndim == 2:
        # A is 3D, B is 2D: broadcast B to all pages
        n, m, p = A.shape
        m2, k = B.shape
        if m != m2:
            raise ValueError(f"Matrix dimensions must match: A.shape={A.shape}, B.shape={B.shape}")
        C = np.zeros((n, k, p))
        for i in range(p):
            C[:, :, i] = A[:, :, i] @ B
        return C
    
    raise ValueError(f"Unsupported array dimensions: A.ndim={A.ndim}, B.ndim={B.ndim}")
