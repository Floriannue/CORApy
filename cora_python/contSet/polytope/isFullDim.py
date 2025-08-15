"""
isFullDim - checks if the dimension of the affine hull of a polytope is
    equal to the dimension of its ambient space; additionally, one can
    obtain a basis of the subspace in which the polytope is contained

Syntax:
    res = isFullDim(P)
    res = isFullDim(P, tol)
    res, subspace = isFullDim(P)

Inputs:
    P - polytope object
    tol - tolerance

Outputs:
    res - true/false
    subspace - (optional) Returns a set of orthogonal unit vectors
               x_1,...,x_k such that P is strictly contained in
               center(P)+span(x_1,...,x_k)
               (here, 'strictly' means that k is minimal).
               Note that if P is just a point, subspace=[].

Authors:       Niklas Kochdumper, Viktor Kotsev, Mark Wetzlinger, Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       02-January-2020 (MATLAB)
Last update:   10-July-2024 (MW, refactor, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional, Union

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.polytope.private.priv_isFullDim_V import priv_isFullDim_V
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog


def isFullDim(P: 'Polytope', tol: float = 1e-12, return_subspace: bool = False) -> Union[bool, Tuple[bool, Optional[np.ndarray]]]:
    """
    Checks if the dimension of the affine hull of a polytope is equal to the dimension of its ambient space.
    
    Args:
        P: polytope object
        tol: tolerance (default: 1e-12)
        
    Returns:
        res: True/False whether polytope is full-dimensional
        subspace: (optional) orthogonal unit vectors defining the subspace
    """
    
    # check if fullDim property is already set (MATLAB: ~isempty(P.fullDim.val))
    if hasattr(P, '_fullDim_val') and P._fullDim_val is not None:
        if return_subspace:
            return bool(P._fullDim_val), (np.eye(P.dim()) if P._fullDim_val else None)
        return bool(P._fullDim_val)
    
    # check whether V- or H-representation given
    if hasattr(P, 'isVRep') and P.isVRep:
        # --- V representation
        # compute degeneracy acc. to [1, (17)] and subspace acc. to [1, (20)]
        res, subspace = priv_isFullDim_V(P.V, tol)
    else:
        # --- H representation
        res, subspace = _aux_isFullDim_Hpoly(P, tol, return_subspace)
    
    # save the set property (only done once, namely, here!)
    P._fullDim_val = bool(res)
    
    if return_subspace:
        # For full-dimensional sets, return identity basis; otherwise whatever was computed
        if res and subspace is None:
            subspace = np.eye(P.dim())
        return res, subspace
    return bool(res)


def _aux_isFullDim_Hpoly(P: 'Polytope', tol: float, return_subspace: bool) -> Tuple[bool, Optional[np.ndarray]]:
    """Auxiliary function for H-representation polytopes (MATLAB translation)."""

    n = P.dim()

    if n == 1:
        # 1D case (MATLAB): rely on dedicated helper, no None branches
        return _aux_isFullDim_1D_Hpoly(P, tol)
    else:
        if not return_subspace:
            # Fast LP-based check (Algorithm [1,(18)])
            return _aux_isFullDim_nD_Hpoly_nosubspace(P, tol), None
        else:
            # Subspace computation (Algorithm [1, Alg. 2])
            return _aux_isFullDim_nD_Hpoly_subspace(P, tol)


def _aux_isFullDim_1D_Hpoly(P: 'Polytope', tol: float) -> Tuple[bool, Optional[np.ndarray]]:
    """1D case for H-representation"""
    # In 1D: determine if set has positive length (full-dim), a single point (not full-dim),
    # unbounded line/half-line (full-dim), or empty (not full-dim).
    A = P.A; b = P.b; Ae = P.Ae; be = P.be
    # Ensure column-vector shape for offsets to avoid indexing errors
    if b is not None and b.ndim == 1:
        b = b.reshape(-1, 1)
    if be is not None and be.ndim == 1:
        be = be.reshape(-1, 1)
    # No constraints at all -> fullspace
    if A.size == 0 and Ae.size == 0:
        P._emptySet_val = False
        P._fullDim_val = True
        return True, None
    # Equality constraints present
    if Ae.size > 0:
        # Any zero-row with nonzero be -> empty
        zero_row = np.allclose(Ae, 0, atol=tol)
        if zero_row and be.size > 0 and not np.allclose(be, 0, atol=tol):
            P._emptySet_val = True
            return False, np.zeros((1, 0))
        # If there are nonzero equalities, check implied value consistency
        if not zero_row:
            implied_vals = []
            for i in range(Ae.shape[0]):
                ae = float(Ae[i, 0])
                be_i = float(be[i, 0]) if be.ndim > 1 else float(be[i])
                if abs(ae) > tol:
                    implied_vals.append(be_i / ae)
                elif abs(be_i) > tol:
                    # 0*x = nonzero -> empty
                    P._emptySet_val = True
                    return False, np.zeros((1, 0))
            if len(implied_vals) > 1:
                first_val = implied_vals[0]
                for v in implied_vals[1:]:
                    if not np.isclose(v, first_val, atol=tol):
                        P._emptySet_val = True
                        return False, np.zeros((1, 0))
            # Nonzero equalities consistent -> fixes x to a point -> not full-dim
            P._emptySet_val = False
            return False, np.zeros((1, 0))
    # Handle inequalities to detect bounds
    upper = np.inf
    lower = -np.inf
    if A.size > 0:
        for i in range(A.shape[0]):
            a = float(A[i, 0])
            bi = float(b[i, 0])
            if a > tol:
                upper = min(upper, bi / a)
            elif a < -tol:
                lower = max(lower, bi / a)
            else:
                if bi < -tol:
                    P._emptySet_val = True
                    return False, np.zeros((1, 0))
    # Apply equality effects if they fixed value (Ae nonzero handled above)
    if np.isfinite(lower) and np.isfinite(upper):
        if upper < lower - tol:
            P._emptySet_val = True
            return False, np.zeros((1, 0))
        if abs(upper - lower) <= tol:
            P._emptySet_val = False
            return False, np.zeros((1, 0))
        return True, None
    # Unbounded in at least one direction -> still full-dim in 1D
    return True, None


def _aux_isFullDim_nD_Hpoly_nosubspace(P: 'Polytope', tol: float) -> bool:
    """nD case for H-representation without subspace (LP-based, MATLAB ref)."""

    # Equality constraints imply degeneracy
    if P.Ae is not None and P.Ae.size > 0:
        return False

    if not P.isHRep:
        P.constraints()

    A = P.A
    b = P.b.reshape(-1, 1) if P.b.ndim == 1 else P.b
    nrCon, n = (A.shape[0], A.shape[1]) if A is not None and A.size > 0 else (0, P.dim())

    # Normalize constraints
    # Compute row norms and normalize A, and adjust b
    if nrCon > 0:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P_A = A / norms
        P_b = b / norms
    else:
        P_A = np.zeros((0, n))
        P_b = np.zeros((0, 1))

    # Quick check for pairwise annihilation ax<=b and -ax<=b2 with -b2>=b
    if P_A.shape[0] > 1:
        dotprod = np.tril(P_A @ P_A.T)
        idx = np.argwhere(withinTol(dotprod, -1))
        for r, c in idx:
            if -P_b[c, 0] > P_b[r, 0] or withinTol(-P_b[c, 0], P_b[r, 0], tol):
                return False

    # Extend A with 2-norm column and set LP to minimize that norm
    A_ext = np.hstack([P_A, np.ones((nrCon, 1))]) if nrCon > 0 else np.zeros((0, n + 1))
    f = np.zeros((n + 1, 1)); f[-1, 0] = -1.0

    problem = {
        'f': f.flatten(),
        'Aineq': A_ext,
        'bineq': P_b.flatten(),
        'Aeq': np.zeros((0, n + 1)),
        'beq': np.zeros(0),
        'lb': np.hstack([np.full(n, -np.inf), 0.0]),
        'ub': None,
    }

    _, r, exitflag, _, _ = CORAlinprog(problem)
    if exitflag == 1:
        # r == 0 => degenerate
        return not withinTol(0, r, 1e-9)
    elif exitflag == -2:
        # empty
        P._emptySet_val = True
        return False
    elif exitflag == -3:
        # numerical/unbounded full-dim
        return True
    elif exitflag < 0:
        raise CORAerror('CORA:solverIssue')
    return True


def _aux_isFullDim_nD_Hpoly_subspace(P: 'Polytope', tol: float) -> Tuple[bool, Optional[np.ndarray]]:
    """Compute (res, subspace) per MATLAB Alg. 2."""
    # Empty -> degenerate
    if P.representsa_('emptySet', tol):
        P._emptySet_val = True
        return False, np.zeros((0, 0))

    n = P.dim()
    # Find feasible point (use center as proxy; if NaN, find via LP of contains(0))
    # For robustness, try origin; if not contained, find any feasible point by minimizing 0 s.t. Ax<=b, Aex=be
    if not P.contains_(np.zeros((n, 1)), 'exact', tol, certToggle=False, scalingToggle=False)[0]:
        # Feasible point via LP: minimize 0 with constraints
        Aeq = P.Ae; beq = P.be
        A = P.A; b = P.b
        # Solve least squares to a feasible point heuristically
        # If equality constraints exist, use them first
        try:
            if Aeq is not None and Aeq.size > 0:
                x0, *_ = np.linalg.lstsq(Aeq, beq, rcond=None)
            else:
                x0 = np.zeros((n, 1))
        except Exception:
            x0 = np.zeros((n, 1))
    else:
        x0 = np.zeros((n, 1))

    # Shift polytope
    P_iter = P - x0.flatten()

    subspace = np.zeros((n, 0))
    for _ in range(n):
        # Find max-norm vector x perpendicular to current subspace directions
        x_iter = _aux_maxNormPerpendicularPolytope(P_iter, subspace)
        if np.linalg.norm(x_iter, ord=np.inf) <= 1e-8:
            break
        x_unit = x_iter / np.linalg.norm(x_iter)
        subspace = np.hstack([subspace, x_unit.reshape(-1, 1)])

    k = subspace.shape[1]
    if k == n:
        return True, np.eye(n)
    else:
        return False, subspace


def _aux_maxNormPerpendicularPolytope(P: 'Polytope', X: np.ndarray) -> np.ndarray:
    """Solve max ||x||_inf s.t. x + X w in P and X' x = 0."""
    n = P.dim()
    if X.size == 0:
        X = np.zeros((n, 0))

    A = P.A; b = P.b
    Ae = P.Ae; be = P.be
    A_ext = np.hstack([A, A @ X]) if A is not None and A.size > 0 else np.zeros((0, n + X.shape[1]))
    Ae_ext = np.hstack([Ae, Ae @ X]) if Ae is not None and Ae.size > 0 else np.zeros((0, n + X.shape[1]))
    y_candidates = []
    for i in range(n):
        e = np.zeros((n, 1)); e[i, 0] = 1.0
        y_candidates.append(e)
        y_candidates.append(-e)

    maximum = 0.0
    maximizer = np.zeros((n, 1))
    for y in y_candidates:
        y_ext = np.vstack([y, np.zeros((X.shape[1], 1))])
        # Add orthogonality constraints X' x = 0
        Ae_full = np.vstack([Ae_ext, np.hstack([X.T, np.zeros((X.shape[1], X.shape[1]))])]) if X.shape[1] > 0 else Ae_ext
        be_full = np.vstack([be, np.zeros((X.shape[1], 1))]) if (be is not None and be.size > 0) else np.zeros((Ae_full.shape[0], 1))

        problem = {
            'f': -y_ext.flatten(),
            'Aineq': A_ext,
            'bineq': b.flatten(),
            'Aeq': Ae_full,
            'beq': be_full.flatten(),
            'lb': None,
            'ub': None,
        }
        x, res, exitflag, _, _ = CORAlinprog(problem)
        if exitflag == -3:  # unbounded
            # enforce y' x = 1
            Aeq2 = np.vstack([Ae_full, y_ext.T])
            beq2 = np.hstack([be_full.flatten(), 1.0])
            problem['Aeq'] = Aeq2
            problem['beq'] = beq2
            x, _, _, _, _ = CORAlinprog(problem)
            res = -np.inf
        if -res > maximum:
            maximum = -res
            maximizer = x[:n].reshape(-1, 1)

    return maximizer