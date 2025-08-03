"""
polytope - converts a zonotope object to a polytope object

Syntax:
    P = polytope(Z)
    P = polytope(Z,method)
    [P,comb,isDeg] = polytope(Z,'exact')

Inputs:
    Z - zonotope object
    method - approximation:
               'exact': based on Theorem 7 of [1]
               'outer:tight': uses interval outer-approximation
               'outer:volume' uses volume

Outputs:
    P - polytope object
    comb - (only method = 'exact') generator combinations corresponding to
               the halfspaces
    isDeg - (only method = 'exact') true/false whether polytope is
               full-dimensional

Example: 
    Z = zonotope([1;-1],[3 2 -1; 1 -2 2]);
    P = polytope(Z);

References:
    [1] Althoff, M.; Stursberg, O. & Buss, M. Computing Reachable Sets
        of Hybrid Systems Using a Combination of Zonotopes and Polytopes
        Nonlinear Analysis: Hybrid Systems, 2010, 4, 233-249

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Niklas Kochdumper, Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       06-August-2018 (MATLAB)
Last update:   10-November-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values

def polytope(Z: Zonotope, method: str = 'exact') -> Union['Polytope', Tuple['Polytope', Optional[np.ndarray], bool]]:
    """
    Converts a zonotope object to a polytope object
    
    Args:
        Z: zonotope object
        method: approximation method:
               'exact': based on Theorem 7 of [1]
               'outer:tight': uses interval outer-approximation
               'outer:volume': uses volume
        
    Returns:
        Polytope object, or tuple (Polytope, comb, isDeg) for exact method
        
    References:
        [1] Althoff, M.; Stursberg, O. & Buss, M. Computing Reachable Sets
            of Hybrid Systems Using a Combination of Zonotopes and Polytopes
            Nonlinear Analysis: Hybrid Systems, 2010, 4, 233-249
    """
    from ..polytope import Polytope
    
    # Fix a tolerance
    tol = 1e-12
    
    # Parse input arguments (matching MATLAB)
    method_values, _ = set_default_values(['exact'], [method])
    method = method_values[0]
    
    # Check input arguments (matching MATLAB)
    inputArgsCheck([[Z, 'att', 'zonotope'],
                   [method, 'str', ['exact', 'outer:tight', 'outer:volume']]])
    
    if method == 'exact':
        # Note: this method was previously called 'polytope'
        
        # Obtain number of generators, dimensions
        Z_compact = Z.compact_('zeros', np.finfo(float).eps)
        comb = None
        isDeg = not _isFullDim(Z_compact, tol)
        
        if Z_compact.G.shape[1] == 0:
            # Generate equality constraint for the center vector
            n = Z_compact.dim()
            C = np.vstack([np.eye(n), -np.eye(n)])
            d = np.vstack([Z_compact.c, -Z_compact.c]).flatten()
        elif not isDeg:
            C, d, comb = aux_polytope_fullDim(Z_compact)
        else:
            C, d = aux_polytope_degenerate(Z_compact, tol)
        
        # Instantiate polytope
        P = Polytope(C, d)
        
        # Polytope is definitely bounded (zonotopes are always bounded)
        P._bounded = True
        
        # Return only polytope for default behavior (matching MATLAB)
        # When called as polytope(Z), return only P
        # When called as polytope(Z, 'exact'), return only P
        # Only when explicitly requested with multiple return values, return tuple
        return P
        
    elif method.startswith('outer'):
        P = aux_polytope_outer(Z, method)
        
        # Polytope is definitely bounded (zonotopes are always bounded)
        P._bounded = True
        
        return P
    else:
        raise CORAerror('CORA:wrongInput', f"Unknown method: {method}")


def _isFullDim(Z: Zonotope, tol: float = 1e-12) -> bool:
    """Check if zonotope is full-dimensional"""
    if Z.G.shape[1] == 0:
        return True
    # Check rank of generator matrix
    rank = np.linalg.matrix_rank(Z.G, tol=tol)
    return rank == Z.dim()


def aux_polytope_fullDim(Z: Zonotope) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Convert full-dimensional zonotope to polytope"""
    c = Z.c
    G = Z.G
    n, nrGen = G.shape
    
    if n == 1:
        # 1D case
        C = np.array([[1], [-1]])
        d = np.array([c.item() + np.sum(np.abs(G)), 
                     -c.item() + np.sum(np.abs(G))])
        return C, d, None
    
    # Get all combinations of n-1 generators for facets (matching MATLAB)
    from cora_python.g.functions.matlab.validate.check.auxiliary.combinator import combinator
    comb = combinator(nrGen, n-1, 'c')
    # Bypass bug in combinator (rows with all-zeros)
    comb = comb[np.any(comb, axis=1), :]
    # Sort combinations to ensure consistent ordering (matching MATLAB)
    comb = comb[np.lexsort(comb.T[::-1])]
    nrComb = comb.shape[0]
    
    if nrComb == 0:
        # Not enough generators
        C = np.array([[1], [-1]])
        d = np.array([c.item(), -c.item()])
        return C, d, None
    
    # Build C matrices for inequality constraint C*x <= d
    C = np.zeros((nrComb, n))
    for i in range(nrComb):
        # Compute n-dimensional cross product with each combination
        # Convert to 0-based indexing for Python
        G_subset = G[:, comb[i, :] - 1]
        C[i, :] = ndimCross(G_subset).flatten()
    
    # Normalize each normal vector (matching MATLAB exactly)
    # MATLAB: C = C ./ vecnorm(C',2,1)';
    # vecnorm(C',2,1)' computes the 2-norm of each column of C' (each row of C)
    # and then transposes the result
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    # Ensure we don't divide by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    C = C / norms
    
    # Remove NaN rows due to rank deficiency
    valid_rows = ~np.any(np.isnan(C), axis=1)
    C = C[valid_rows, :]
    comb = comb[valid_rows, :] if comb.size > 0 else None
    
    if C.shape[0] == 0:
        # Fallback to interval-based polytope
        I = Z.interval()
        C = np.vstack([np.eye(n), -np.eye(n)])
        d = np.hstack([I.sup.flatten(), -I.inf.flatten()])
        return C, d, None
    
    # Determine offset vector
    deltaD = np.sum(np.abs(C @ G), axis=1)
    
    # Construct the overall inequality constraints (matching MATLAB order)
    # MATLAB: d = [C*c + deltaD; -C*c + deltaD]; C = [C; -C];
    d_positive = C @ c.flatten() + deltaD
    d_negative = -C @ c.flatten() + deltaD
    d = np.hstack([d_positive, d_negative])
    C = np.vstack([C, -C])
    
    return C, d, comb


def ndimCross(vectors: np.ndarray) -> np.ndarray:
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
        # General n-dimensional case using determinant (matching MATLAB exactly)
        # MATLAB: v(i,1) = (-1)^(i+1) * det(Q([1:i-1,i+1:end],:));
        result = np.zeros(n)
        for i in range(n):
            # Create submatrix by removing i-th row
            rows_to_keep = list(range(i)) + list(range(i+1, n))
            submatrix = vectors[rows_to_keep, :]
            
            # Compute determinant with alternating sign (matching MATLAB)
            result[i] = ((-1) ** (i + 1)) * np.linalg.det(submatrix)
        return result


def aux_polytope_degenerate(Z: Zonotope, tol: float) -> Tuple[np.ndarray, np.ndarray]:
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
            C_proj, d_proj, _ = aux_polytope_fullDim(Z_proj)
            
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
        C, d, _ = aux_polytope_fullDim(Z)
    
    return C, d


def aux_polytope_outer(Z: Zonotope, method: str) -> 'Polytope':
    """Compute outer-approximating polytope"""
    from ..polytope import Polytope
    
    if method == 'outer:tight':
        # Axis-aligned approximation
        I = Z.interval()
        Z_red = I.zonotope()
        P = Z_red.polytope('exact')[0]  # Get first return value
        
        # PCA-based approximation
        Z_red2 = Z.reduce('pca')
        Z_red2 = aux_repair(Z_red2, Z)
        Padd = Z_red2.polytope('exact')[0]  # Get first return value
        
        # Intersect results
        P = P.and_(Padd)
        
    elif method == 'outer:volume':
        # PCA-based approximation
        Z_red1 = Z.reduce('pca')
        Z_red1 = aux_repair(Z_red1, Z)
        vol1 = Z_red1.volume_('exact')
        
        # Axis-aligned approximation
        I = Z.interval()
        Z_red2 = I.zonotope()
        Z_red2 = aux_repair(Z_red2, Z)
        vol2 = Z_red2.volume_('exact')
        
        if vol1 < vol2:
            P = Z_red1.polytope('exact')[0]  # Get first return value
        else:
            P = Z_red2.polytope('exact')[0]  # Get first return value
    else:
        raise CORAerror('CORA:wrongInput', f"Unknown outer method: {method}")
    
    return P


def aux_repair(Z: Zonotope, Zorig: Zonotope) -> Zonotope:
    """
    Repair zonotope if there is no length in one dimension
    
    Args:
        Z: zonotope to repair
        Zorig: original zonotope for reference
        
    Returns:
        Repaired zonotope
    """
    # Get length of each dimension
    I = Z.interval()
    len_dims = 2 * I.rad()
    
    # Find zero lengths
    index = np.where(len_dims == 0)[0]
    
    if len(index) > 0:
        # Construct zonotope to be added
        I_orig = Zorig.interval()
        orig_len = 2 * I_orig.rad()
        orig_center = I_orig.center()
        
        # Get Z matrix
        Zmat = np.hstack([Z.c, Z.G])
        
        for i in range(len(index)):
            ind = index[i]
            # Replace center
            Zmat[ind, 0] = orig_center[ind]
            # Replace generator value
            Zmat[ind, ind + 1] = 0.5 * orig_len[ind]
        
        # Instantiate zonotope
        Zrep = Zonotope(Zmat[:, 0:1], Zmat[:, 1:])
    else:
        Zrep = Z
    
    return Zrep 