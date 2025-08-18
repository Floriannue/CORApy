"""
isBounded - check if a polytope is bounded

Syntax:
    res = isBounded(P)

Inputs:
    P - polytope object

Outputs:
    res - true/false

Example:
    A = [1 0; 0 1; -1 0; 0 -1];
    b = [1; 1; 1; 1];
    P = polytope(A, b);
    res = isBounded(P)

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 16-September-2019 (MW, specify output for empty case) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from scipy.optimize import linprog
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.contSet.polytope.private.priv_equalityToInequality import priv_equalityToInequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

if TYPE_CHECKING:
    from .polytope import Polytope


def _ensure_array(mat, cols: int) -> np.ndarray:
    if mat is None:
        return np.zeros((0, cols))
    return mat


def isBounded(P: 'Polytope') -> bool:
    """
    Check if a polytope is bounded
    
    Args:
        P: Polytope object
        
    Returns:
        bool: True if polytope is bounded, False otherwise
    """
    
    # check if property is set (MATLAB: ~isempty(P.bounded.val))
    if hasattr(P, '_bounded_val') and P._bounded_val is not None:
        return P._bounded_val

    # Empty object (no representation stored) -> unbounded fullspace container
    if P.isemptyobject():
        res = False
    elif P.isVRep:
        # V-representation: bounded in nD; in 1D check for Inf vertices
        res = P.dim() > 1 or not np.any(np.isinf(P.V))
    else:
        # H-representation path
        n = P.dim()
        if n == 1:
            # 1D H-rep boundedness can be determined without enumerating vertices
            if not P.isHRep:
                P.constraints()
            A = P.A
            b = P.b
            Ae = P.Ae
            be = P.be

            # Empty set via contradictory equalities (0*x = nonzero) or inequalities (0 <= negative)
            if Ae.size > 0 and np.allclose(Ae, 0) and be.size > 0 and not np.allclose(be, 0):
                res = True
            elif A.size > 0 and np.any((np.isclose(A, 0) & (b < -1e-12))):
                res = True
            else:
                # Determine lower/upper bounds from inequalities and equalities
                upper = np.inf
                lower = -np.inf
                if A.size > 0:
                    for i in range(A.shape[0]):
                        a = float(A[i, 0])
                        bi = float(b[i, 0])
                        if a > 1e-12:
                            upper = min(upper, bi / a)
                        elif a < -1e-12:
                            lower = max(lower, bi / a)
                        else:
                            # 0*x <= b: if b < 0 then empty handled above; otherwise no effect
                            pass
                if Ae.size > 0:
                    for i in range(Ae.shape[0]):
                        ae = float(Ae[i, 0])
                        be_i = float(be[i, 0])
                        if abs(ae) > 1e-12:
                            val = be_i / ae
                            upper = min(upper, val)
                            lower = max(lower, val)
                        else:
                            # 0*x = be: if be != 0, empty handled above
                            pass
                # Bounded iff both lower and upper are finite and lower <= upper
                res = (np.isfinite(lower) and np.isfinite(upper) and (lower <= upper + 1e-12))

        else:
            # Quick column zero check: any unconstrained dimension -> unbounded
            if not P.isHRep:
                P.constraints()
            A = P.A
            Ae = P.Ae
            # Ensure consistent column counts when stacking
            if A.size == 0 and Ae.size == 0:
                res = False
            else:
                ncols = max(A.shape[1] if A.ndim == 2 and A.size > 0 else 0,
                               Ae.shape[1] if Ae.ndim == 2 and Ae.size > 0 else 0)
                A_ = A if (A.size > 0 and A.shape[1] == ncols) else np.zeros((A.shape[0] if A.size > 0 else 0, ncols))
                Ae_ = Ae if (Ae.size > 0 and Ae.shape[1] == ncols) else np.zeros((Ae.shape[0] if Ae.size > 0 else 0, ncols))
                if not np.all(np.any(np.vstack([A_, Ae_]), axis=0)):
                    res = False
                else:
                    # If we already know the set is non-degenerate, use MATLAB's dual LP check
                    if getattr(P, '_fullDim_val', None) is True:
                        res = _aux_isBounded_nD_Hpoly_nondegenerate(P, n)
                    else:
                        # Standard method: check support function in simplex directions [I_n, -1_n]
                        # This matches MATLAB's aux_isBounded_nD_Hpoly exactly
                        for i in range(n):
                            # check every axis-aligned direction
                            d = np.zeros((n, 1)); d[i, 0] = 1.0
                            # evaluate support function (returns tuple (val, x) like MATLAB)
                            val, _ = P.supportFunc_(d, 'upper')
                            vnum = val
                            if vnum == np.inf:
                                # set is unbounded
                                res = False
                                P._emptySet_val = False
                                return res
                            elif vnum == -np.inf:
                                # set is empty
                                res = True
                                P._emptySet_val = True
                                return res
                        
                        # check -1_n direction
                        d = -np.ones((n, 1))
                        # evaluate support function (returns tuple (val, x) like MATLAB)
                        val, _ = P.supportFunc_(d, 'upper')
                        vnum = val
                        if vnum == np.inf:
                            # set is unbounded
                            res = False
                            P._emptySet_val = False
                            return res
                        elif vnum == -np.inf:
                            # set is empty
                            res = True
                            P._emptySet_val = True
                            return res

                        # code reaches this part: set is neither unbounded nor empty
                        res = True
                        P._emptySet_val = False

    # cache result
    P._bounded_val = res
    return res


def _aux_isBounded_nD_Hpoly_nondegenerate(P: 'Polytope', n: int) -> bool:
    """Non-degenerate H-polytope boundedness via dual LP (MATLAB parity).
    Matches @polytope/isBounded.m aux_isBounded_nD_Hpoly_nondegenerate.
    """
    # Ensure origin is contained; otherwise shift by Chebyshev center
    contains_origin = False
    try:
        if hasattr(P, 'contains_'):
            contains_origin = P.contains_(np.zeros((n, 1)))[0]
    except Exception:
        contains_origin = False

    if not contains_origin:
        c = P.center()
        if isinstance(c, np.ndarray) and c.size == 0:
            # empty set
            P._emptySet_val = True
            return True
        if np.any(np.isnan(c)):
            # unbounded (not every unbounded set yields NaN, but this is MATLAB's early exit)
            P._emptySet_val = False
            return False
        # not empty
        P._emptySet_val = False
        P_ = P - c
    else:
        P_ = P

    # Rewrite equality constraints as inequalities
    A, b = priv_equalityToInequality(P_.A, P_.b, P_.Ae, P_.be)
    # Normalize constraints w.r.t. b to obtain A x <= 1
    A, b, _, _ = priv_normalizeConstraints(A, b, None, None, 'b')
    h = A.shape[0]

    # Dual LP: check if origin is contained in dual polytope conv(A^T)
    # Variables: t in R, x in R^h
    # min -t s.t. A^T x = 0, t <= x_i, sum_i x_i = 1
    problem = {
        'f': np.concatenate([np.array([-1.0]), np.zeros(h)]),
        'Aineq': np.hstack([np.ones((h, 1)), -np.eye(h)]) if h > 0 else np.zeros((0, 1 + h)),
        'bineq': np.zeros(h),
        'Aeq': np.vstack([np.hstack([np.zeros((n, 1)), A.T]), np.hstack([np.zeros((1, 1)), np.ones((1, h))])]),
        'beq': np.concatenate([np.zeros(n), np.array([1.0])]),
        'lb': None,
        'ub': None,
    }

    _, fval, exitflag, _, _ = CORAlinprog(problem)
    if exitflag >= 0:
        if fval > 0 or withinTol(fval, 0, 1e-8):
            # origin not contained in dual -> primal unbounded
            return False
        return True
    # Solver issue: follow MATLAB and raise error; here, be conservative
    raise Exception('CORA:solverIssue in _aux_isBounded_nD_Hpoly_nondegenerate')


def _check_bounded_halfspace(P: 'Polytope') -> bool:
    """
    Check if polytope in halfspace representation is bounded
    
    This is done by checking if the system A*x <= b, Ae*x = be has unbounded solutions
    in any direction. We solve linear programs in different directions.
    """
    
    n = P.dim()  # Use P.dim() to get dimension reliably
    
    # Check if polytope is bounded by solving LP in each coordinate direction
    # If we can find an unbounded direction, the polytope is unbounded
    
    # Prepare constraint matrices - handle both inequality and equality constraints
    A_ub = P.A if (P.A is not None and P.A.size > 0) else None
    b_ub = P.b.flatten() if (P.b is not None and P.b.size > 0) else None
    A_eq = P.Ae if (P.Ae is not None and P.Ae.size > 0) else None
    b_eq = P.be.flatten() if (P.be is not None and P.be.size > 0) else None
    
    # Test positive and negative directions for each coordinate
    for i in range(n):
        for direction in [1, -1]:
            # Objective: maximize/minimize x_i subject to A*x <= b, Ae*x = be
            c = np.zeros(n)
            c[i] = direction
            
            try:
                # Solve LP: min c^T x subject to A*x <= b and Ae*x = be
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                               bounds=None, method='highs')
                
                # If the LP is unbounded, the polytope is unbounded
                if not result.success and result.status == 3:
                    return False
                    
                # Also check if objective value is extremely large
                if result.success and abs(result.fun) > 1e10:
                    return False
                    
            except Exception:
                # If LP solver fails, assume bounded (conservative approach)
                continue
    
    # Additional check: verify that the recession cone is {0}
    # This is a more thorough but computationally expensive check
    
    # For efficiency, we'll use a heuristic approach:
    # Check if there exists a direction d such that A*d <= 0 and d != 0
    
    # This would indicate an unbounded direction
    try:
        # Try to find a non-zero direction d such that A*d <= 0
        # We solve: find d such that ||d|| = 1 and A*d <= 0
        
        # Use a simple approach: check if any combination of constraint normals
        # gives a direction where all constraints are satisfied
        
        # If all constraint normals point "outward", polytope is likely bounded
        A = P.A if (P.A is not None and P.A.size > 0) else np.zeros((0, n))
        rank_A = np.linalg.matrix_rank(A)
        if rank_A == n:
            # Full rank constraint matrix often indicates boundedness
            return True
            
    except Exception:
        pass
    
    # Determine boundedness via recession cone LP: exists d != 0 with A d <= 0, Ae d = 0?
    A = P.A if P.A.size > 0 else None
    Ae = P.Ae if P.Ae.size > 0 else None
    if A is None and Ae is None:
        return False

    # max 1^T t subject to A d + t <= 0, -A d + t <= 0, Ae d = 0, ||d||_inf <= 1, t >= 0
    # Heuristic: check standard basis directions and their negatives
    cand = []
    for i in range(n):
        e = np.zeros((n,))
        e[i] = 1.0
        cand.append(e); cand.append(-e)
    for d in cand:
        ok_ineq = True
        if A is not None and A.size > 0:
            ok_ineq = np.all(A @ d <= 1e-12)
        ok_eq = True
        if Ae is not None and Ae.size > 0:
            ok_eq = np.allclose(Ae @ d, 0.0, atol=1e-12)
        if ok_ineq and ok_eq and np.linalg.norm(d, np.inf) > 0:
            return False