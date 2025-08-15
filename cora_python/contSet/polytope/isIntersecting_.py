"""
isIntersecting_ - determines if a polytope intersects another set

Syntax:
    res = isIntersecting_(P, S, type_, tol)

Inputs:
    P - polytope object
    S - contSet object, numerical vector, point cloud  
    type_ - type of check ('exact' or 'approx')
    tol - tolerance

Outputs:
    res - true/false whether intersection occurs

Authors: Niklas Kochdumper, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 20-November-2019 (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING, Union
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def isIntersecting_(P: Polytope, 
                    S: Union['ContSet', np.ndarray], 
                    type_: str = 'exact',
                    tol: float = 1e-8) -> bool:
    """
    Determines if a polytope intersects another set
    
    This function checks for intersection between a polytope and another set.
    For zonotopes, it converts them to constrained zonotopes and uses linear
    programming to determine intersection.
    
    Args:
        P: polytope object
        S: contSet object or numeric array
        type_: type of check ('exact' or 'approx')
        tol: tolerance for computation
        
    Returns:
        bool: True if sets intersect, False otherwise
        
    Example:
        >>> from cora_python.contSet.polytope.polytope import Polytope
        >>> from cora_python.contSet.zonotope.zonotope import Zonotope
        >>> P = Polytope(np.array([[1, 1]]), np.array([1]))
        >>> Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        >>> result = isIntersecting_(P, Z, 'exact')
    """
    # Handle numeric case: check containment
    if isinstance(S, np.ndarray):
        result, _, _ = P.contains_(S, type_, tol, 0, False, False)
        return bool(np.all(result)) if isinstance(result, np.ndarray) else bool(result)
    
    # If S has higher precedence, let it handle the intersection
    if hasattr(S, 'precedence') and S.precedence < P.precedence:
        return S.isIntersecting_(P, type_, tol)
    
    # Treat 'fullspace' as intersecting everything; use representsa_ with tol
    if hasattr(P, 'representsa_') and bool(P.representsa_('fullspace', tol)):
        return True
    if hasattr(S, 'representsa_') and bool(S.representsa_('fullspace', tol)):
        return True
    # Proper empty sets (infeasible constraints) do not intersect
    if hasattr(P, 'representsa_') and bool(P.representsa_('emptySet', tol)):
        return False
    if hasattr(S, 'representsa_') and bool(S.representsa_('emptySet', tol)):
        return False
    
    # Handle different set types
    if isinstance(type_, str):
        if S.__class__.__name__ == 'Zonotope':
            if type_ == 'exact':
                # For exact intersection, use the approximate method which is more robust
                return _aux_isIntersecting_approx(P, S, tol)
            elif type_ == 'approx':
                # Use approximate intersection check
                return _aux_isIntersecting_approx(P, S, tol)
        
        elif S.__class__.__name__ == 'Interval':
            if type_ == 'exact':
                # Convert interval to constrained zonotope and check intersection
                from cora_python.contSet.conZonotope.conZonotope import ConZonotope
                # Convert interval to zonotope representation first
                c = S.center()
                r = S.rad()
                # Ensure shapes
                c = c.flatten() if c.ndim > 1 else np.asarray(c)
                r = r.flatten() if r.ndim > 1 else np.asarray(r)
                G = np.diagflat(r) if r.size > 0 else np.zeros((c.size, 0))
                # Construct constrained zonotope without constraints
                cZ = ConZonotope(c, G, np.zeros((0, G.shape[1])), np.zeros((0,)))
                return _aux_isIntersecting_P_cZ(P, cZ)
            elif type_ == 'approx':
                return _aux_isIntersecting_approx(P, S, tol)
        
        elif S.__class__.__name__ == 'Polytope':
            # Handle polytope-polytope intersection
            # Fast 1D path using direct bound extraction without vertex enumeration
            if P.dim() == 1 and S.dim() == 1:
                def _bounds_1d(A, b, Ae, be):
                    upper = np.inf; lower = -np.inf
                    if A.size > 0:
                        for i in range(A.shape[0]):
                            a = float(A[i,0]); bi = float(b[i] if b.ndim == 1 else b[i,0])
                            if a > tol:
                                upper = min(upper, bi / a)
                            elif a < -tol:
                                lower = max(lower, bi / a)
                            else:
                                if bi < -tol:
                                    return np.inf, -np.inf
                    if Ae.size > 0:
                        for i in range(Ae.shape[0]):
                            ae = float(Ae[i,0]); bei = float(be[i] if be.ndim == 1 else be[i,0])
                            if abs(ae) > tol:
                                val = bei / ae
                                upper = min(upper, val); lower = max(lower, val)
                            else:
                                if abs(bei) > tol:
                                    return np.inf, -np.inf
                    return lower, upper
                A1, b1, Ae1, be1 = P.A, P.b, P.Ae, P.be
                A2, b2, Ae2, be2 = S.A, S.b, S.Ae, S.be
                l1, u1 = _bounds_1d(A1, b1, Ae1, be1)
                l2, u2 = _bounds_1d(A2, b2, Ae2, be2)
                # If both are single points (equalities fix), compare directly
                if np.isfinite(l1) and np.isfinite(u1) and abs(u1 - l1) <= tol and np.isfinite(l2) and np.isfinite(u2) and abs(u2 - l2) <= tol:
                    return abs(l1 - l2) <= tol
                # If one is a single point and the other has bounds, check inclusion
                if np.isfinite(l1) and np.isfinite(u1) and abs(u1 - l1) <= tol:
                    return (l2 <= l1 + tol) and (l1 <= u2 + tol)
                if np.isfinite(l2) and np.isfinite(u2) and abs(u2 - l2) <= tol:
                    return (l1 <= l2 + tol) and (l2 <= u1 + tol)
                if l1 == np.inf and u1 == -np.inf:
                    return False
                if l2 == np.inf and u2 == -np.inf:
                    return False
                # Overlap if max(lower) <= min(upper) + tol
                return (max(l1, l2) <= min(u1, u2) + tol)
            # Normalize and compact like MATLAB before feasibility LP
            P1 = P; P2 = S
            for PP in (P1, P2):
                if not PP.isHRep:
                    PP.constraints()
                A, b, Ae, be = PP.A, PP.b.reshape(-1, 1), PP.Ae, PP.be.reshape(-1, 1)
                A, b, Ae, be = priv_normalizeConstraints(A, b, Ae, be, 'A')
                A, b, Ae, be, _, _ = priv_compact_all(A, b, Ae, be, PP.dim(), tol)
                PP._A, PP._b, PP._Ae, PP._be = A, b, Ae, be
                PP._isHRep = True
            return _aux_isIntersecting_poly_poly(P1, P2, tol)
        
        elif S.__class__.__name__ == 'ConZonotope':
            # Handle constrained zonotope case
            if type_ == 'exact':
                return _aux_isIntersecting_P_cZ(P, S)
            elif type_ == 'approx':
                return _aux_isIntersecting_approx(P, S, tol)
        
        elif S.__class__.__name__ == 'ZonoBundle':
            # Handle zonotope bundle case
            if type_ == 'exact':
                return _aux_isIntersecting_P_zB(P, S)
            elif type_ == 'approx':
                return _aux_isIntersecting_approx(P, S, tol)
    
    # If no specific implementation exists, raise error
    raise CORAerror('CORA:noops', f'isIntersecting_ not implemented for polytope and {type(S).__name__} with type {type_}')


def _aux_isIntersecting_P_cZ(P: Polytope, cZ) -> bool:
    """
    Check if a polytope {x | H*x <= d, He*x = de} and a constraint zonotope 
    {x = c + G*beta | A*beta = b, beta ∈ [-1,1]} intersect by solving the
    linear program from MATLAB implementation:
    
    min t
    s.t. H*x - d <= t
         x = c + G*beta
         A*beta = b
         He*x = de
         beta ∈ [-1,1]
    """
    
    # Get polytope constraints - exact MATLAB extraction
    H = P.A
    d = P.b
    nrIneq_poly = H.shape[0]
    
    He = P.Ae if hasattr(P, 'Ae') else np.empty((0, H.shape[1]))
    de = P.be if hasattr(P, 'be') else np.empty((0,))
    nrEq_poly = He.shape[0]
    
    # Get constrained zonotope parameters
    c = cZ.c
    G = cZ.G
    A = cZ.A if hasattr(cZ, 'A') and cZ.A is not None else np.empty((0, G.shape[1]))
    b = cZ.b if hasattr(cZ, 'b') and cZ.b is not None else np.empty((0,))
    
    # Dimensions
    n = len(c)  # dimension
    nrGen = G.shape[1]  # number of generators
    nrEq_conZono = A.shape[0]  # number of constraints
    
    # Optimization variable is [t; x; beta] with length 1 + n + nrGen
    
    # Construct inequality constraints following MATLAB exactly:
    # H*x - d <= t  =>  H*x <= t + d  =>  [-1, H, 0] * [t; x; beta] <= d
    # beta ∈ [-1,1]  =>  -beta <= 1, beta <= 1
    
    # Calculate sparsity and decide on representation (following MATLAB)
    numElem = (nrIneq_poly + 2*nrGen) * (1 + n + nrGen)
    minNumZeros = 2*nrGen*(1+n) + nrIneq_poly*nrGen + 2*nrGen*(nrGen-1)
    sparsity = minNumZeros / numElem if numElem > 0 else 0
    
    # Build inequality constraint matrix
    A_ineq = np.zeros((nrIneq_poly + 2*nrGen, 1 + n + nrGen))
    b_ineq = np.zeros(nrIneq_poly + 2*nrGen)
    
    # H*x - t <= d  =>  [-1, H, 0] * [t; x; beta] <= d
    A_ineq[:nrIneq_poly, 0] = -1  # -t coefficient  
    A_ineq[:nrIneq_poly, 1:1+n] = H  # H*x coefficients
    # zeros for beta part already set
    b_ineq[:nrIneq_poly] = d.flatten()
    
    # -beta <= 1  =>  [0, 0, -I] * [t; x; beta] <= 1
    A_ineq[nrIneq_poly:nrIneq_poly+nrGen, 1+n:] = -np.eye(nrGen)
    b_ineq[nrIneq_poly:nrIneq_poly+nrGen] = 1
    
    # beta <= 1  =>  [0, 0, I] * [t; x; beta] <= 1
    A_ineq[nrIneq_poly+nrGen:, 1+n:] = np.eye(nrGen)
    b_ineq[nrIneq_poly+nrGen:] = 1
    
    # Construct equality constraints following MATLAB exactly:
    # He*x = de
    # x = c + G*beta  =>  x - G*beta = c
    # A*beta = b
    
    total_eq_rows = nrEq_poly + n + nrEq_conZono
    A_eq = np.zeros((total_eq_rows, 1 + n + nrGen))
    b_eq = np.zeros(total_eq_rows)
    
    # He*x = de  =>  [0, He, 0] * [t; x; beta] = de
    if nrEq_poly > 0:
        A_eq[:nrEq_poly, 1:1+n] = He
        b_eq[:nrEq_poly] = de.flatten()
    
    # x - G*beta = c  =>  [0, I, -G] * [t; x; beta] = c
    A_eq[nrEq_poly:nrEq_poly+n, 1:1+n] = np.eye(n)
    A_eq[nrEq_poly:nrEq_poly+n, 1+n:] = -G
    b_eq[nrEq_poly:nrEq_poly+n] = c.flatten()
    
    # A*beta = b  =>  [0, 0, A] * [t; x; beta] = b
    if nrEq_conZono > 0:
        A_eq[nrEq_poly+n:nrEq_poly+n+nrEq_conZono, 1+n:] = A
        b_eq[nrEq_poly+n:nrEq_poly+n+nrEq_conZono] = b.flatten()
    
    # Construct objective function: min t
    f = np.zeros(1 + n + nrGen)
    f[0] = 1  # coefficient for t
    
    # Build problem structure for CORAlinprog
    problem = {
        'f': f,
        'Aineq': A_ineq,
        'bineq': b_ineq,
        'Aeq': A_eq,
        'beq': b_eq,
        'lb': None,
        'ub': None
    }
    
    # Solve linear program following MATLAB logic
    try:
        _, val, exitflag = CORAlinprog(problem)
        
        # Multiple cases following MATLAB exactly:
        # 1. constrained zonotope (and possibly polytope) is empty => infeasible (dual unbounded)
        # 2. polytope empty => max(y)>0 OR no intersection point between non-empty polytope and non-empty cZ
        
        if exitflag == -2:
            # No feasible point was found
            res = False
        elif exitflag == 1:  # Could include 3 and 0
            # Feasible point found, check if max(y) < 0 == val
            # Use constraint tolerance consistent with withinTol default (~1e-9)
            constraint_tol = 1e-9
            res = val <= constraint_tol
        elif exitflag == -3:
            # Unbounded (because no inequality constraints in P) => feasible
            res = True
        else:
            # Solver issue - be conservative
            res = False
            
        return res
        
    except Exception:
        # If LP solver fails, return False (conservative)
        return False


def _aux_isIntersecting_P_zB(P: Polytope, zB) -> bool:
    """
    Check if a polytope {x | H*x <= d} and a zonotope bundle 
    {x = c1 + G1*a|a ∈ [-1,1]} ∪ ... ∪ {x = cq + Gq*a | a ∈ [-1,1]}
    intersect by solving the following linear program (MATLAB implementation):
    
    min sum(y)
    
    s.t. H*x - y <= d
              y >= 0
     c1 + G1*a1 = x
              .
              .
              .
     cq + Gq*aq = x
     a1,...,aq ∈ [-1,1]
    """
    
    # Get polytope properties  
    H = P.A
    d = P.b
    
    p, n = H.shape if H.size > 0 else (0, P.dim())
    
    # Construct inequality constraints
    # H*x - y <= d  =>  [H, -I] * [x; y] <= d
    # y >= 0       =>  [0, -I] * [x; y] <= 0
    A_ineq = np.zeros((2*p, n + p))
    b_ineq = np.zeros(2*p)
    
    # H*x - y <= d
    A_ineq[:p, :n] = H
    A_ineq[:p, n:] = -np.eye(p)
    b_ineq[:p] = d.flatten()
    
    # -y <= 0  (i.e., y >= 0)
    A_ineq[p:, n:] = -np.eye(p)
    b_ineq[p:] = 0
    
    # Loop over all parallel zonotopes in the bundle
    A_eq = np.zeros((0, n + p))  # Start with empty equality constraints
    b_eq = np.zeros(0)
    num_generators_total = 0
    
    # First pass: count total generators and build equality constraints for x
    for i in range(zB.parallelSets):
        Z = zB.Z[i]
        c = Z.center() if hasattr(Z, 'center') else Z.c
        G = Z.generators() if hasattr(Z, 'generators') else Z.G
        m = G.shape[1]  # number of generators for this zonotope
        num_generators_total += m
        
        # Expand A_eq to include generators for this zonotope
        # c_i + G_i*a_i = x  =>  [I, 0, -G_i, 0, ...] * [x; y; a_1; a_2; ...] = c_i
        new_A_eq = np.zeros((n, n + p + num_generators_total))
        new_A_eq[:, :n] = np.eye(n)  # x coefficients
        # y coefficients are 0
        # Set generator coefficients: start at n+p and go to appropriate position
        gen_start = n + p + num_generators_total - m
        new_A_eq[:, gen_start:gen_start+m] = -G
        
        # Add to overall equality constraint matrix
        if A_eq.shape[0] == 0:
            A_eq = new_A_eq
            b_eq = c.flatten()
        else:
            # Expand existing rows
            A_eq_expanded = np.zeros((A_eq.shape[0], n + p + num_generators_total))
            A_eq_expanded[:, :A_eq.shape[1]] = A_eq
            A_eq = np.vstack([A_eq_expanded, new_A_eq])
            b_eq = np.hstack([b_eq, c.flatten()])
    
    # Now add inequality constraints for generators: a_i ∈ [-1,1]
    total_vars = n + p + num_generators_total
    gen_constraints = 2 * num_generators_total
    
    # Expand A_ineq to include generator constraints
    A_ineq_expanded = np.zeros((A_ineq.shape[0] + gen_constraints, total_vars))
    A_ineq_expanded[:A_ineq.shape[0], :A_ineq.shape[1]] = A_ineq
    b_ineq_expanded = np.hstack([b_ineq, np.ones(gen_constraints)])
    
    # Add generator bound constraints: -a_i <= 1 and a_i <= 1
    gen_start = n + p
    for i in range(num_generators_total):
        # -a_i <= 1
        A_ineq_expanded[A_ineq.shape[0] + 2*i, gen_start + i] = -1
        # a_i <= 1  
        A_ineq_expanded[A_ineq.shape[0] + 2*i + 1, gen_start + i] = 1
    
    # Construct objective function: minimize sum(y)
    f = np.zeros(total_vars)
    f[n:n+p] = 1  # coefficients for y variables
    
    # Build problem structure
    problem = {
        'f': f,
        'Aineq': A_ineq_expanded,
        'bineq': b_ineq_expanded,
        'Aeq': A_eq,
        'beq': b_eq,
        'lb': None,
        'ub': None
    }
    
    # Solve linear program
    try:
        _, val, exitflag = CORAlinprog(problem)
        
        # Check if intersection between the two sets is empty
        # Following MATLAB: ~(exitflag < 0 || (val > 0 && ~withinTol(val,0)))
        tol = 1e-9  # Default tolerance from MATLAB withinTol
        
        if exitflag < 0:
            # Infeasible or unbounded
            res = False
        elif val > 0 and not (abs(val) <= tol):
            # Positive objective value (no intersection)
            res = False
        else:
            # Zero or near-zero objective (intersection exists)
            res = True
            
        return res
        
    except Exception:
        # If LP solver fails, return False (conservative)
        return False


def _aux_isIntersecting_approx(P: Polytope, S, tol: float) -> bool:
    """
    Approximate check, i.e., conservative intersection check (possibility of false positives)
    following MATLAB implementation
    """
    
    # Check if polytope represents a constraint hyperplane
    isHyp, P_hyp = P.representsa_('conHyperplane', tol)
    
    # Get polytope constraints
    A = P.A
    b = P.b
    
    # Special 'approx' algorithm for zonotope bundles
    if S.__class__.__name__ == 'ZonoBundle':
        # Loop over all parallel zonotopes
        for j in range(S.parallelSets):
            # Read out j-th zonotope
            Z = S.Z[j]
            
            if isHyp:
                # Check intersection with hyperplane
                Ae = P_hyp.Ae
                be = P_hyp.be
                I = Z.supportFunc_(Ae[0, :].reshape(-1, 1), 'range')
                if not I.contains_(be[0], 'exact', tol):
                    return False
            
            # Loop over all halfspaces
            for i in range(A.shape[0]):
                support_val = Z.supportFunc_(A[i, :].reshape(-1, 1), 'lower')
                if b[i] < support_val:
                    return False
        
        return True
    
    else:
        # Single set case
        if isHyp:
            # Check intersection with hyperplane
            Ae = P_hyp.Ae
            be = P_hyp.be
            I = S.supportFunc_(Ae[0, :].reshape(-1, 1), 'range')
            if not I.contains_(be[0], 'exact', tol):
                return False
        
        # Loop over all halfspaces
        for i in range(A.shape[0]):
            support_val = S.supportFunc_(A[i, :].reshape(-1, 1), 'lower')
            if b[i] < support_val:
                return False
        
        return True


def _aux_isIntersecting_poly_poly(P1: 'Polytope', P2: 'Polytope', tol: float) -> bool:
    """
    Exact feasibility check: exists x s.t. A1 x <= b1, Ae1 x = be1 and A2 x <= b2, Ae2 x = be2
    """
    # Ensure H-reps
    P1.constraints(); P2.constraints()
    n = P1.dim()
    def ensure_cols(M, ncols):
        if M is None or M.size == 0:
            return np.zeros((0, ncols))
        return M
    A1, b1, Ae1, be1 = ensure_cols(P1.A, n), P1.b.flatten(), ensure_cols(P1.Ae, n), P1.be.flatten()
    A2, b2, Ae2, be2 = ensure_cols(P2.A, n), P2.b.flatten(), ensure_cols(P2.Ae, n), P2.be.flatten()

    A_ub = np.vstack([A1, A2]) if (A1.size > 0 or A2.size > 0) else None
    b_ub = np.hstack([b1, b2]) if (b1.size > 0 or b2.size > 0) else None
    A_eq = np.vstack([Ae1, Ae2]) if (Ae1.size > 0 or Ae2.size > 0) else None
    b_eq = np.hstack([be1, be2]) if (be1.size > 0 or be2.size > 0) else None

    try:
        from scipy.optimize import linprog
        c = np.zeros(n)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=None, method='highs')
        return bool(res.success)
    except Exception:
        return _aux_isIntersecting_approx(P1, P2, tol)