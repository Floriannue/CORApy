"""
zonotope - converts a zonotope object to a polytope object

This function converts a zonotope to a polytope using exact conversion
based on halfspace representation.

Syntax:
    P = zonotope(Z)
    P = zonotope(Z, method)

Inputs:
    Z - zonotope object
    method - approximation method:
               'exact': based on Theorem 7 of [1] (default)
               'outer:tight': uses interval outer-approximation
               'outer:volume' uses volume

Outputs:
    P - polytope object

Example: 
    Z = zonotope([1;-1],[3 2 -1; 1 -2 2]);
    P = polytope(Z);

References:
   [1] Althoff, M.; Stursberg, O. & Buss, M. Computing Reachable Sets
       of Hybrid Systems Using a Combination of Zonotopes and Polytopes
       Nonlinear Analysis: Hybrid Systems, 2010, 4, 233-249

Authors: Niklas Kochdumper, Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 06-August-2018 (MATLAB)
Last update: 10-November-2022 (MW, unify with other polytope functions) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Tuple, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.auxiliary import combinator
from cora_python.g.functions.helper import ndimCross


def zonotope(Z, method: str = 'exact'):
    """
    Convert a zonotope object to a polytope object
    
    Args:
        Z: Zonotope object
        method: Conversion method ('exact', 'outer:tight', 'outer:volume')
        
    Returns:
        Polytope object
    """
    # Import here to avoid circular imports
    from .polytope import Polytope
    
    # fix a tolerance
    tol = 1e-12

    # Validate input
    if not hasattr(Z, 'c') or not hasattr(Z, 'G'):
        raise CORAerror('CORA:wrongInputInConstructor',
                       'Input must be a zonotope object')

    if method not in ['exact', 'outer:tight', 'outer:volume']:
        raise CORAerror('CORA:wrongInput',
                       'Invalid method. Must be exact, outer:tight, or outer:volume')

    if method == 'exact':
        # Compact zonotope (remove zero generators)
        G = Z.G[:, np.any(Z.G != 0, axis=0)]
        c = Z.c
        
        # Check if zonotope is degenerate (not full dimensional)
        n, nrGen = G.shape
        isDeg = not _is_full_dim(Z, tol)

        if nrGen == 0:
            # Generate equality constraint for the center vector
            A = np.vstack([np.eye(n), -np.eye(n)])
            b = np.vstack([c, -c])
        elif not isDeg:
            A, b = _polytope_full_dim(c, G)
        else:
            A, b = _polytope_degenerate(c, G, tol)

        # Instantiate polytope
        P = Polytope(A, b)

    elif method.startswith('outer'):
        P = _polytope_outer(Z, method)

    # Polytope is definitely bounded (zonotopes are always bounded)
    P.bounded = True

    return P


def _is_full_dim(Z, tol: float) -> bool:
    """Check if zonotope is full dimensional"""
    if Z.G.size == 0:
        return False
    
    # Check rank of generator matrix
    rank = np.linalg.matrix_rank(Z.G, tol=tol)
    return rank == Z.G.shape[0]


def _polytope_full_dim(c: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert full-dimensional zonotope to polytope constraints"""
    n, nrGen = G.shape

    if n == 1:
        # 1D case
        A = np.array([[1], [-1]])
        deltaD = np.sum(np.abs(G))
        c_scalar = c.item() if c.size == 1 else c[0, 0]  # Extract scalar value
        b = np.array([[c_scalar + deltaD], [-c_scalar + deltaD]])
        return A, b
    else:
        # Get number of possible facets
        comb = combinator(nrGen, n-1, 'c')
        # Remove rows with all zeros (bypass bug in combinator)
        comb = comb[np.any(comb, axis=1), :]
        nrComb = comb.shape[0]

        # Build C matrices for inequality constraint C*x < d
        C = np.zeros((nrComb, n))
        for i in range(nrComb):
            # Compute n-dimensional cross product with each combination
            generators = G[:, comb[i, :] - 1]  # Convert to 0-based indexing
            C[i, :] = ndimCross(generators).flatten()
        
        # Normalize each normal vector
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        # Avoid division by zero
        valid_norms = norms.flatten() > 1e-12
        C[valid_norms, :] = C[valid_norms, :] / norms[valid_norms, :]

        # Remove NaN rows due to rank deficiency
        valid_rows = ~np.any(np.isnan(C), axis=1)
        C = C[valid_rows, :]

        # Determine offset vector in addition to center
        deltaD = np.sum(np.abs(C @ G), axis=1, keepdims=True)
         
        # Construct the overall inequality constraints
        A = np.vstack([C, -C])
        b = np.vstack([C @ c + deltaD, -C @ c + deltaD])

        return A, b


def _polytope_degenerate(c: np.ndarray, G: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert degenerate (rank-deficient) zonotope to polytope constraints"""
    n, nrGen = G.shape

    # Singular value decomposition
    U, S, _ = np.linalg.svd(G, full_matrices=True)
    
    # Pad S to correct size if needed
    if S.size < n:
        S_full = np.zeros(n)
        S_full[:S.size] = S
        S = S_full

    # State space transformation
    Z_transformed = U.T @ np.column_stack([c, G])

    # Remove dimensions with all zeros
    ind = np.where(S <= tol)[0]
    ind_ = np.setdiff1d(np.arange(S.size), ind)

    if len(ind) > 0:
        # Import here to avoid circular imports
        from cora_python.contSet.zonotope import Zonotope
        
        # Compute polytope in transformed space
        c_reduced = Z_transformed[ind_, 0:1]
        G_reduced = Z_transformed[ind_, 1:]
        Z_reduced = Zonotope(c_reduced, G_reduced)
        P_reduced = zonotope(Z_reduced)

        # Transform back to original space
        A_padded = np.hstack([P_reduced.A, np.zeros((P_reduced.A.shape[0], len(ind)))])
        A = A_padded @ U.T
        b = P_reduced.b

        # Add equality constraint restricting polytope to null-space
        U_null = U[:, ind]
        A_eq = np.vstack([U_null.T, -U_null.T])
        b_eq = np.vstack([U_null.T @ c, -U_null.T @ c])
        
        A = np.vstack([A, A_eq])
        b = np.vstack([b, b_eq])

    return A, b


def _polytope_outer(Z, method: str):
    """Compute outer approximation of zonotope by polytope"""
    # Import here to avoid circular imports
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.zonotope import Zonotope
    
    if method == 'outer:tight':
        # Solution 1 (axis-aligned): convert to interval then to polytope
        I = Interval(Z)
        Z_red = Zonotope(I)
        P1 = zonotope(Z_red, 'exact')
        
        # Solution 2 (method C): reduce zonotope using PCA
        Z_red2 = Z.reduce('pca')
        Z_red2 = _repair_zonotope(Z_red2, Z)
        P2 = zonotope(Z_red2, 'exact')
        
        # Intersect results
        P = P1 & P2
        
    elif method == 'outer:volume':
        # Solution 1 (method C): reduce using PCA
        Z_red1 = Z.reduce('pca')
        Z_red1 = _repair_zonotope(Z_red1, Z)
        vol1 = Z_red1.volume()
        
        # Solution 2 (axis-aligned): convert to interval
        I = Interval(Z)
        Z_red2 = Zonotope(I)
        Z_red2 = _repair_zonotope(Z_red2, Z)
        vol2 = Z_red2.volume()

        if vol1 < vol2:
            P = zonotope(Z_red1, 'exact')
        else:
            P = zonotope(Z_red2, 'exact')
    
    return P


def _repair_zonotope(Z_red, Z_orig):
    """Repair reduced zonotope to ensure it encloses original"""
    # This is a placeholder for the repair function
    # In practice, this would ensure the reduced zonotope still encloses the original
    return Z_red 