import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contSet.polytope.private.priv_equalityToInequality import priv_equalityToInequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope


def vertices_(P: 'Polytope', method: str = 'lcon2vert') -> np.ndarray:
    """
    Computes the vertices of a polytope.
    This is a Python translation of the MATLAB CORA implementation.
    """
    tol = 1e-12

    # If polytope has V-representation, return it (assume minimal)
    if P.isVRep:
        return P.V

    n = P.dim()

    # Check if polytope is known to be empty
    if P.isemptyobject():
        V = np.zeros((n, 0))
        P._V = V
        P._isVRep = True
        P._minVRep_val = True
        P._emptySet_val = True
        P._bounded_val = True
        P._fullDim_val = False
        return V

    # 1D case quick
    if n == 1:
        # Convert to H-rep if not already
        if not P.isHRep:
            P.constraints()
        V = _priv_vertices_1D(P.A, P.b, P.Ae, P.be, tol)
        # Unbounded signalled by +/-Inf among candidates
        if V.size > 0 and np.any(np.isinf(V)):
            P._emptySet_val = False
            P._bounded_val = False
            raise CORAerror('CORA:notSupported', 'Vertex enumeration requires a bounded polytope.')
        # Empty set
        if V.size == 0:
            V_out = np.zeros((1, 0))
            P._V = V_out
            P._isVRep = True
            P._minVRep_val = True
            P._emptySet_val = True
            P._bounded_val = True
            P._fullDim_val = False
            return V_out
        # Bounded 1D
        P._V = V
        P._isVRep = True
        P._emptySet_val = False
        P._bounded_val = True
        P._fullDim_val = V.shape[1] > 1
        return P.V

    # Detect unboundedness early
    if not P.isemptyobject() and not P.isBounded():
        P._emptySet_val = False
        P._bounded_val = False
        raise CORAerror('CORA:notSupported', 'Vertex enumeration requires a bounded polytope.')

    # Compute Chebyshev center to detect empty cases
    c = P.center()
    if np.any(np.isnan(c)):
        # Treat NaN in center as unbounded/invalid
        P._emptySet_val = False
        P._bounded_val = False
        raise CORAerror('CORA:notSupported', 'Vertex enumeration requires a bounded polytope.')
    elif (isinstance(c, np.ndarray) and c.size == 0) or (isinstance(c, float) and np.isnan(c)):
        # Empty
        V = np.zeros((n, 0))
        P._V = V
        P._isVRep = True
        P._minVRep_val = True
        P._emptySet_val = True
        P._bounded_val = True
        P._fullDim_val = False
        return V

    # Ensure H-representation is available for methods
    if not P.isHRep:
        P.constraints()

    if method == 'lcon2vert':
        V = _aux_vertices_lcon2vert(P, n, c)
    elif method == 'comb':
        V = _aux_vertices_comb(P)
    else:
        raise ValueError(f"Invalid method '{method}'. Allowed: 'lcon2vert', 'comb'.")

    # Set properties after computation
    P._V = V
    P._isVRep = True
    P._minVRep_val = True
    P._emptySet_val = False
    P._bounded_val = True
    # Determine degeneracy via SVD
    if V.shape[1] <= V.shape[0]:
        P._fullDim_val = False
    else:
        Vc = V - np.mean(V, axis=1, keepdims=True)
        _, S, _ = np.linalg.svd(Vc)
        P._fullDim_val = n == np.count_nonzero(~withinTol(S, 0, 1e-12))

    return V


def _priv_vertices_1D(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray, tol: float) -> np.ndarray:
    """Auxiliary function for 1D case, extracting vertices from H-rep.
    Matches MATLAB's priv_vertices_1D.
    """
    # Collect all boundary points from inequalities and equalities
    boundaries = []

    # From inequalities: A*x <= b
    # For A[i,0] > 0, x <= b[i,0]/A[i,0] (upper bound)
    # For A[i,0] < 0, x >= b[i,0]/A[i,0] (lower bound, divide by negative)
    upper_bounds = []
    lower_bounds = []

    if A.size > 0:
        for i in range(A.shape[0]):
            if A[i, 0] > tol: # Check for positive coefficient
                upper_bounds.append(b[i, 0] / A[i, 0])
            elif A[i, 0] < -tol: # Check for negative coefficient
                lower_bounds.append(b[i, 0] / A[i, 0])
            elif withinTol(A[i, 0], 0, tol) and b[i, 0] < -tol: # 0*x <= negative -> empty set
                return np.zeros((1, 0))

    # From equalities: Ae*x == be
    if Ae.size > 0:
        for i in range(Ae.shape[0]):
            if not withinTol(Ae[i, 0], 0, tol):
                val = be[i, 0] / Ae[i, 0]
                upper_bounds.append(val)
                lower_bounds.append(val)
            elif withinTol(Ae[i, 0], 0, tol) and not withinTol(be[i, 0], 0, tol):
                return np.zeros((1, 0))

    min_val = -np.inf if not lower_bounds else np.max(lower_bounds)
    max_val = np.inf if not upper_bounds else np.min(upper_bounds)

    if min_val > max_val + tol:
        return np.zeros((1, 0))

    if withinTol(min_val, max_val, tol):
        return np.array([[min_val]])

    if not np.isinf(min_val) and not np.isinf(max_val):
        return np.array([[min_val, max_val]])
    elif not np.isinf(min_val) and np.isinf(max_val):
        # Unbounded above -> return finite and +Inf to signal unboundedness
        return np.array([[min_val, np.inf]])
    elif np.isinf(min_val) and not np.isinf(max_val):
        # Unbounded below -> return -Inf and finite
        return np.array([[-np.inf, max_val]])
    else:
        # Fully unbounded in 1D -> return empty to allow upstream to treat as unbounded
        return np.zeros((1, 0))


def _compute_affine_subspace_from_active_set(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray, c: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Compute an orthonormal basis S of the affine hull of the polytope at center c.
    Uses active constraints at c (inequalities at equality within tol and all equalities).
    Returns S with shape (n, k). If k==0, polytope is a single point.
    """
    n = c.shape[0]
    rows = []
    if A.size > 0:
        # Active inequalities where A_i c ~= b_i
        act = np.where(np.abs(A @ c - b) <= tol)[0]
        if act.size > 0:
            rows.append(A[act, :])
    if Ae.size > 0:
        rows.append(Ae)
    if len(rows) == 0:
        # No active constraints -> full space
        return np.eye(n)
    M = np.vstack(rows)
    # S is null space of M
    # Compute nullspace via SVD
    U, s, Vt = np.linalg.svd(M)
    rank = np.sum(s > 1e-12)
    S = Vt.T[:, rank:]
    return S


def _aux_vertices_lcon2vert(P: 'Polytope', n: int, c: np.ndarray) -> np.ndarray:
    """
    Vertex enumeration using a duality-like approach with degeneracy handling based on
    affine subspace computed from active constraints at the center.
    """
    A = P.A
    b = P.b.flatten()
    Ae = P.Ae
    be = P.be.flatten()

    halfspaces = []
    for i in range(A.shape[0]):
        halfspaces.append(np.hstack([A[i, :], -b[i]]))
    for i in range(Ae.shape[0]):
        halfspaces.append(np.hstack([Ae[i, :], -be[i]]))
        halfspaces.append(np.hstack([-Ae[i, :], be[i]]))
    halfspaces = np.array(halfspaces, dtype=float)

    def _handle_degeneracy() -> np.ndarray:
        # Compute affine subspace basis S from active constraints
        S = _compute_affine_subspace_from_active_set(A, b.reshape(-1, 1), Ae, be.reshape(-1, 1), c)
        k = S.shape[1]
        if k == 0:
            return c.reshape(-1, 1)
        if 0 < k < n:
            A_sub = A @ S
            b_sub = b.reshape(-1, 1) - (A @ c).reshape(-1, 1) if A.size > 0 else b.reshape(-1, 1)
            Ae_sub = Ae @ S if Ae.size > 0 else np.zeros((0, k))
            be_sub = be.reshape(-1, 1) - (Ae @ c).reshape(-1, 1) if Ae.size > 0 else np.zeros((0, 1))
            from cora_python.contSet.polytope.polytope import Polytope
            P_sub = Polytope(A_sub, b_sub, Ae_sub, be_sub)
            V_sub = vertices_(P_sub, 'lcon2vert')
            return S @ V_sub + c
        # k == n but we are here -> likely numeric issue or unbounded; guard boundedness
        if not P.isBounded():
            raise CORAerror('CORA:notSupported', 'Vertex enumeration requires a bounded polytope.')
        # Fallback
        return _aux_vertices_comb(P)

    if halfspaces.shape[0] == 0:
        return _handle_degeneracy()

    c_pt = c.flatten()
    if c_pt.size != n:
        c_pt = np.zeros(n)

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point=c_pt)
        V_pts = hs.intersections
        if V_pts.size == 0:
            return _handle_degeneracy()
        tol_local = 1e-10
        keep = np.ones(V_pts.shape[0], dtype=bool)
        if A.shape[0] > 0:
            # Compare each candidate against all A rows; broadcast b correctly per candidate
            AV = A @ V_pts.T  # shape (m, k)
            keep &= np.all(AV.T <= (b.reshape(-1, 1) + tol_local).T, axis=1)
        if Ae.shape[0] > 0:
            AVe = Ae @ V_pts.T
            keep &= np.all(np.abs(AVe.T - be.reshape(1, -1)) <= 1e-8, axis=1)
        V_pts = V_pts[keep, :]
        if V_pts.shape[0] == 0:
            return _handle_degeneracy()
        V_round = np.round(V_pts / tol_local) * tol_local
        _, unique_idx = np.unique(V_round, axis=0, return_index=True)
        V_pts = V_pts[np.sort(unique_idx), :]
        V_sorted = V_pts[np.lexsort(tuple((-V_pts[:, j] for j in range(V_pts.shape[1]-1, -1, -1)))), :]
        return V_sorted.T
    except Exception:
        # If center is not strictly interior (common for unbounded), fall back
        return _handle_degeneracy()


def _aux_vertices_comb(P: 'Polytope') -> np.ndarray:
    """
    Simple vertex enumeration algorithm: returns a superset containing the true vertices.
    Mirrors MATLAB's aux_vertices_comb; may return extra points.
    """
    tol = 1e-12
    n = P.dim()

    A_orig = P.A
    b_orig = P.b
    Ae_orig = P.Ae
    be_orig = P.be

    A, b = priv_equalityToInequality(A_orig, b_orig, Ae_orig, be_orig)
    A, b, _, _ = priv_normalizeConstraints(A, b, np.array([[]]).reshape(0, n), np.array([[]]).reshape(0, 1), 'A')

    A, b, _, _, empty, minHRep = priv_compact_all(A, b, np.array([[]]).reshape(0, 0), np.array([[]]).reshape(0, 1), P.dim(), tol)
    if empty:
        return np.zeros((P.dim(), 0))

    nrCon, n = A.shape
    if n > nrCon:
        raise CORAerror('CORA:notSupported', "Method 'comb' does not support degenerate cases.")

    from cora_python.g.functions.matlab.validate.check.auxiliary import combinator
    comb = combinator(nrCon, n, 'c')
    nrComb = comb.shape[0]
    if nrComb > 10000:
        raise CORAerror('CORA:specialError', 'Too many combinations.')

    V = np.zeros((n, nrComb))
    idxKeep = np.ones(nrComb, dtype=bool)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='A_eq does not appear to be of full row rank.')
        for i in range(nrComb):
            indices = comb[i, :] - 1
            A_sub = A[list(indices), :]
            b_sub = b[list(indices)]
            if np.linalg.matrix_rank(A_sub, tol=1e-8) < n:
                idxKeep[i] = False
                continue
            try:
                V[:, i] = np.linalg.solve(A_sub, b_sub).flatten()
            except Exception:
                idxKeep[i] = False
                continue

    V = V[:, idxKeep]

    V_round = np.round(V / tol) * tol
    _, unique_idx = np.unique(V_round, axis=1, return_index=True)
    V = V[:, np.sort(unique_idx)]

    keep2 = np.ones(V.shape[1], dtype=bool)
    if A.shape[0] > 0:
        keep2 &= np.all(A @ V <= b + 1e-10, axis=0)
    # Equalities were converted to inequalities above; no separate Ae/be filtering needed here
    return V[:, keep2] 