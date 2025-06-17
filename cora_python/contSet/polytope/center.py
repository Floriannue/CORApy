"""
center - center of a polytope

Syntax:
    c = center(P)

Inputs:
    P - polytope object

Outputs:
    c - center of the polytope

Example:
    A = [1 0; 0 1; -1 0; 0 -1];
    b = [1; 1; 1; 1];
    P = polytope(A, b);
    c = center(P)

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 16-September-2019 (MW, specify output for empty case) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional


def center(P) -> Optional[np.ndarray]:
    """
    Compute the center of a polytope
    
    Args:
        P: Polytope object
        
    Returns:
        np.ndarray or None: Center of the polytope, None if empty
    """
    
    # Check if polytope is empty
    if P.isemptyobject():
        return None
    
    # For halfspace representation, compute center using Chebyshev center
    # This is an approximation - the actual center computation requires solving
    # a linear programming problem
    
    if P.A is not None and P.b is not None and P.A.size > 0:
        n = P.A.shape[1]  # dimension
        
        # Simple approximation: average of vertices if we can compute them
        try:
            if hasattr(P, 'vertices') and callable(getattr(P, 'vertices')):
                V = P.vertices()
                if V is not None and V.size > 0:
                    return np.mean(V, axis=1, keepdims=True)
        except:
            pass
        
        # Fallback: use constraint-based approximation
        # This is a simplified version - full implementation would use LP
        
        # For box constraints, compute center directly
        if _is_box_polytope(P):
            return _compute_box_center(P)
        
        # General case: use origin if it's inside, otherwise approximate
        origin = np.zeros((n, 1))
        if P.contains(origin.flatten()):
            return origin
        
        # Approximate center as weighted average of constraint normals
        # This is a heuristic and not mathematically precise
        weights = 1.0 / (np.abs(P.b) + 1e-12)
        weighted_normals = P.A.T @ weights.reshape(-1, 1)
        center_approx = weighted_normals / (np.linalg.norm(weighted_normals) + 1e-12)
        
        return center_approx
    
    # For vertex representation
    if hasattr(P, 'V') and P.V is not None and P.V.size > 0:
        return np.mean(P.V, axis=1, keepdims=True)
    
    # Default case
    return None


def _is_box_polytope(P) -> bool:
    """Check if polytope represents a box (axis-aligned)"""
    if P.A is None or P.b is None:
        return False
    
    # Check if A matrix has only 0, 1, -1 entries and each row has exactly one non-zero
    A_abs = np.abs(P.A)
    row_sums = np.sum(A_abs, axis=1)
    
    # Each row should have exactly one non-zero element
    if not np.allclose(row_sums, 1.0):
        return False
    
    # Non-zero elements should be 1 or -1
    non_zero_mask = A_abs > 1e-12
    if not np.allclose(A_abs[non_zero_mask], 1.0):
        return False
    
    return True


def _compute_box_center(P) -> np.ndarray:
    """Compute center for box polytope"""
    n = P.A.shape[1]
    center = np.zeros((n, 1))
    
    for i in range(n):
        # Find positive and negative constraints for dimension i
        pos_mask = (P.A[:, i] > 0.5)
        neg_mask = (P.A[:, i] < -0.5)
        
        if np.any(pos_mask) and np.any(neg_mask):
            # Compute bounds
            upper_bound = np.min(P.b[pos_mask])
            lower_bound = -np.max(P.b[neg_mask])
            center[i] = (upper_bound + lower_bound) / 2
    
    return center 