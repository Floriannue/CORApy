"""
zonotope - converts a polytope object to a zonotope object (Polytope -> Zonotope)

This function converts a polytope to a zonotope. For bounded polytopes we
produce an axis-aligned zonotope using the interval hull; for 1D we return the
exact zonotope of the interval; for empty polytopes we return an empty
zonotope of the correct dimension; for unbounded polytopes we raise an error
to align with MATLAB behavior used in tests.

Syntax:
    Z = zonotope(P)
    Z = zonotope(P, method)

Inputs:
    P - polytope object (H- or V-representation)
    method - approximation method (currently only 'exact' supported for 1D and
             axis-aligned interval hull for higher dimensions)

Outputs:
    Z - zonotope object

Authors: Test-driven Python translation by AI Assistant (based on CORA)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Tuple, List, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from scipy.linalg import sqrtm
from scipy.optimize import linprog

if TYPE_CHECKING:
    from .polytope import Polytope


def zonotope(Z, method: str = 'outer'):
    """
    Convert a polytope object to a zonotope object (Polytope -> Zonotope)
    
    Args:
        Z: Polytope object
        method: 'outer' (default). 'inner' and 'exact' raise not supported.
        
    Returns:
        Zonotope object
    """
    # Import here to avoid circular imports
    from .polytope import Polytope
    from cora_python.contSet.zonotope import Zonotope

    # Tolerance
    tol = 1e-12

    # If input is already a Zonotope, return a copy (allow pass-through)
    if hasattr(Z, 'c') and hasattr(Z, 'G') and isinstance(Z, Zonotope):
        return Zonotope(Z)

    # Require Polytope input
    if not isinstance(Z, Polytope):
        raise CORAerror('CORA:wrongInputInConstructor', 'Input must be a polytope object')

    P = Z
    n = P.dim()

    # Empty polytope -> return empty zonotope of dimension n
    if P.isemptyobject():
        return Zonotope.empty(n)

    # Unbounded polytopes are not supported for conversion
    if not P.isBounded() and n > 0:
        raise CORAerror('CORA:specialError', 'Polytope is unbounded and can therefore not be converted into a zonotope.')

    # Mode handling like MATLAB
    mode = method
    if mode not in ['outer', 'inner', 'exact']:
        raise CORAerror('CORA:wrongInput', 'Invalid method. Must be exact, outer, or inner')

    if mode == 'exact':
        raise CORAerror('CORA:notSupported')

    if mode == 'inner':
        # Inner approximation not yet translated
        raise CORAerror('CORA:notSupported')

    # outer mode below

    # 1D outer conversion (exact interval)
    if n == 1:
        # Get vertices (min/max)
        V = P.vertices_().flatten()
        if V.size == 0 or np.any(np.isnan(V)):
            # Treat as empty
            return Zonotope.empty(1)
        vmin = np.min(V)
        vmax = np.max(V)
        c = np.array([[0.5 * (vmin + vmax)]])
        G = np.array([[0.5 * (vmax - vmin)]])
        return Zonotope(c, G)

    # For higher dimensions: refined outer approximation using MVE preconditioning
    # 1) Compute inner ellipsoid E of P (approximate), center c_e and shape Q
    try:
        E = P.ellipsoid(mode='inner')
        c_e = E.q  # (n,1)
        Q = E.Q    # (n,n)
        # Numerical guard: ensure symmetric PSD
        Q = 0.5 * (Q + Q.T)
        Q_r = sqrtm(Q).real
        # If sqrtm failed or Q_r singular, fall back to interval hull
        if not np.all(np.isfinite(Q_r)):
            raise ValueError('sqrtm failed')
    except Exception:
        Q_r = None

    if Q_r is None:
        # Fallback to interval hull (axis-aligned outer)
        return _outer_interval_hull(P)

    # 2) Transform polytope Mt: x = Q_r y + c_e ⇒ A(Q_r y + c_e) <= b
    Mt = Polytope(P)  # copy
    A = P.A; b = P.b; Ae = P.Ae; be = P.be
    if A is None or A.size == 0:
        Mt.A = np.zeros((0, n))
        Mt.b = np.zeros((0, 1))
    else:
        Mt.A = A @ Q_r
        Mt.b = (b - A @ c_e).reshape(-1, 1)
    if Ae is None or Ae.size == 0:
        Mt.Ae = np.zeros((0, n))
        Mt.be = np.zeros((0, 1))
    else:
        Mt.Ae = Ae @ Q_r
        Mt.be = (be - Ae @ c_e).reshape(-1, 1)
    if P.isVRep and P.V.size > 0:
        try:
            Mt.V = np.linalg.pinv(Q_r) @ (P.V - c_e)
        except Exception:
            pass

    # 3) Compute enclosing hyperbox in transformed space via support LPs
    sF_upper = np.zeros((n, 1))
    sF_lower = np.zeros((n, 1))
    I = np.eye(n)
    for i in range(n):
        e = I[:, i]
        s_upper = _support_linear_program(Mt, e, 'upper')
        s_lower = _support_linear_program(Mt, e, 'lower')
        sF_upper[i, 0] = s_upper
        sF_lower[i, 0] = s_lower
    ct = 0.5 * (sF_upper + sF_lower)
    rt = 0.5 * (sF_upper - sF_lower)

    # 4) Back-transform zonotope: Z = Q_r*Zt + c_e
    Gt = np.diagflat(rt)
    c_final = (Q_r @ ct) + c_e
    G_final = Q_r @ Gt
    return Zonotope(c_final, G_final)


def _outer_interval_hull(P: 'Polytope'):
    """Axis-aligned outer approximation by interval hull (fallback)."""
    n = P.dim()
    P.constraints()
    lb = np.zeros((n, 1))
    ub = np.zeros((n, 1))
    A_ub = P.A if P.A.size > 0 else None
    b_ub = P.b.flatten() if P.b.size > 0 else None
    A_eq = P.Ae if P.Ae.size > 0 else None
    b_eq = P.be.flatten() if P.be.size > 0 else None
    for i in range(n):
        c_vec = np.zeros(n); c_vec[i] = 1.0
        res_max = linprog(-c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
        res_min = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
        if not (res_max.success and res_min.success):
            raise CORAerror('CORA:solverIssue', 'Failed to compute interval hull for zonotope conversion')
        ub[i, 0] = -res_max.fun
        lb[i, 0] = res_min.fun
    c = 0.5 * (lb + ub)
    widths = 0.5 * (ub - lb)
    G = np.diagflat(widths)
    from cora_python.contSet.zonotope import Zonotope
    return Zonotope(c, G)


def _support_linear_program(P: 'Polytope', direction: np.ndarray, bound_type: str) -> float:
    """Support function via LP: maximize/minimize direction^T x subject to P."""
    n = P.dim()
    d = direction.flatten()
    c = -d if bound_type == 'upper' else d
    A_ub = P.A if (P.A is not None and P.A.size > 0) else None
    b_ub = P.b.flatten() if (P.b is not None and P.b.size > 0) else None
    A_eq = P.Ae if (P.Ae is not None and P.Ae.size > 0) else None
    b_eq = P.be.flatten() if (P.be is not None and P.be.size > 0) else None
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    if not res.success:
        return np.inf if bound_type == 'upper' else -np.inf
    return (-res.fun) if bound_type == 'upper' else res.fun


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