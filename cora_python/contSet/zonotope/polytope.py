"""
polytope method for zonotope class
"""

import numpy as np
from typing import Tuple, Optional, List
from .zonotope import Zonotope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polytope import Polytope


def polytope(Z: Zonotope, method: str = 'exact') -> 'Polytope':
    """
    Converts a zonotope object to a polytope object
    
    Args:
        Z: zonotope object
        method: approximation method:
               'exact': based on Theorem 7 of [1]
               'outer:tight': uses interval outer-approximation
               'outer:volume': uses volume
        
    Returns:
        Polytope object
        
    References:
        [1] Althoff, M.; Stursberg, O. & Buss, M. Computing Reachable Sets
            of Hybrid Systems Using a Combination of Zonotopes and Polytopes
            Nonlinear Analysis: Hybrid Systems, 2010, 4, 233-249
    """
    from ..polytope import Polytope
    
    # Fix a tolerance
    tol = 1e-12
    
    if method == 'exact':
        # Compact the zonotope (remove zero generators)
        Z_compact = Z.compact_('zeros', np.finfo(float).eps)
        
        # Check if zonotope is full-dimensional
        isDeg = not _isFullDim(Z_compact, tol)
        
        if Z_compact.G.shape[1] == 0:
            # Generate equality constraint for the center vector (point)
            n = Z_compact.dim()
            C = np.vstack([np.eye(n), -np.eye(n)])
            d = np.vstack([Z_compact.c, -Z_compact.c]).flatten()
        elif not isDeg:
            C, d = _polytope_fullDim(Z_compact)
        else:
            C, d = _polytope_degenerate(Z_compact, tol)
        
        # Instantiate polytope
        P = Polytope(C, d)
        
    elif method.startswith('outer'):
        P = _polytope_outer(Z, method)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return P


def _isFullDim(Z: Zonotope, tol: float = 1e-12) -> bool:
    """Check if zonotope is full-dimensional"""
    return Z.rank(tol) == Z.dim()


def _polytope_fullDim(Z: Zonotope) -> Tuple[np.ndarray, np.ndarray]:
    """Convert full-dimensional zonotope to polytope"""
    c = Z.c
    G = Z.G
    n, nrGen = G.shape
    
    if n == 1:
        # 1D case
        C = np.array([[1], [-1]])
        d = np.array([c.item() + np.sum(np.abs(G)), 
                     -c.item() + np.sum(np.abs(G))])
        return C, d
    
    # Get all combinations of n-1 generators for facets
    from itertools import combinations
    comb_list = list(combinations(range(nrGen), n-1))
    
    if not comb_list:
        # Not enough generators
        C = np.array([[1], [-1]])
        d = np.array([c.item(), -c.item()])
        return C, d
    
    # Build C matrices for inequality constraint C*x <= d
    C_list = []
    for comb in comb_list:
        if len(comb) == n-1:
            # Compute normal vector using cross product (for n-1 vectors in n dimensions)
            G_subset = G[:, comb]
            normal = _ndimCross(G_subset)
            if not np.allclose(normal, 0):
                # Normalize
                normal = normal / np.linalg.norm(normal)
                C_list.append(normal)
    
    if not C_list:
        # Fallback to interval-based polytope
        I = Z.interval()
        C = np.vstack([np.eye(n), -np.eye(n)])
        d = np.hstack([I.sup.flatten(), -I.inf.flatten()])
        return C, d
    
    C = np.array(C_list)
    
    # Determine offset vector
    deltaD = np.sum(np.abs(C @ G), axis=1)
    
    # Construct the overall inequality constraints
    d = np.hstack([C @ c.flatten() + deltaD, -C @ c.flatten() + deltaD])
    C = np.vstack([C, -C])
    
    return C, d


def _ndimCross(vectors: np.ndarray) -> np.ndarray:
    """Compute n-dimensional cross product for n-1 vectors in n dimensions"""
    n, k = vectors.shape
    
    if k != n - 1:
        raise ValueError("Need exactly n-1 vectors for n-dimensional cross product")
    
    if n == 2:
        # 2D case: cross product of 1 vector
        v = vectors[:, 0]
        return np.array([-v[1], v[0]])
    elif n == 3:
        # 3D case: cross product of 2 vectors
        v1, v2 = vectors[:, 0], vectors[:, 1]
        return np.cross(v1, v2)
    else:
        # General n-dimensional case using determinant
        # Create matrix with standard basis vectors and input vectors
        result = np.zeros(n)
        for i in range(n):
            # Create matrix with i-th standard basis vector replaced
            M = np.eye(n)
            M[:, :k] = vectors
            # Remove i-th row and compute determinant
            M_reduced = np.delete(M, i, axis=0)
            result[i] = (-1)**i * np.linalg.det(M_reduced)
        return result


def _polytope_degenerate(Z: Zonotope, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert degenerate (not full-dimensional) zonotope to polytope"""
    # For degenerate case, use SVD to find the subspace
    c = Z.c
    G = Z.G
    n, nrGen = G.shape
    
    # Singular value decomposition
    U, S, _ = np.linalg.svd(G, full_matrices=True)
    
    # Find dimensions with significant singular values
    ind_active = np.where(S > tol)[0]
    ind_null = np.where(S <= tol)[0]
    
    if len(ind_null) > 0:
        # Transform to subspace
        Z_proj = Zonotope(U[:, ind_active].T @ c, U[:, ind_active].T @ G)
        
        # Convert projected zonotope to polytope
        if Z_proj.dim() > 0:
            C_proj, d_proj = _polytope_fullDim(Z_proj)
            
            # Transform back to original space
            C = C_proj @ U[:, ind_active].T
            d = d_proj
            
            # Add equality constraints for null space
            if len(ind_null) > 0:
                U_null = U[:, ind_null]
                C_eq = np.vstack([U_null.T, -U_null.T])
                d_eq = np.hstack([U_null.T @ c.flatten(), -U_null.T @ c.flatten()])
                
                C = np.vstack([C, C_eq])
                d = np.hstack([d, d_eq])
        else:
            # Point case
            C = np.vstack([np.eye(n), -np.eye(n)])
            d = np.hstack([c.flatten(), -c.flatten()])
    else:
        # Full rank case
        C, d = _polytope_fullDim(Z)
    
    return C, d


def _polytope_outer(Z: Zonotope, method: str) -> 'Polytope':
    """Compute outer-approximating polytope"""
    from ..polytope import Polytope
    
    if method == 'outer:tight':
        # Axis-aligned approximation
        I = Z.interval()
        Z_red = I.zonotope()
        P1 = Z_red.polytope('exact')
        
        # PCA-based approximation
        Z_red2 = Z.reduce('pca')
        P2 = Z_red2.polytope('exact')
        
        # Intersect results
        P = P1.and_(P2)
        
    elif method == 'outer:volume':
        # PCA-based approximation
        Z_red1 = Z.reduce('pca')
        vol1 = Z_red1.volume_('exact')
        
        # Axis-aligned approximation
        I = Z.interval()
        Z_red2 = I.zonotope()
        vol2 = Z_red2.volume_('exact')
        
        if vol1 < vol2:
            P = Z_red1.polytope('exact')
        else:
            P = Z_red2.polytope('exact')
    else:
        raise ValueError(f"Unknown outer method: {method}")
    
    return P 