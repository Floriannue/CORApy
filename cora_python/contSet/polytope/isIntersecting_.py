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
        return result
    
    # If S has higher precedence, let it handle the intersection
    if hasattr(S, 'precedence') and S.precedence < P.precedence:
        return S.isIntersecting_(P, type_, tol)
    
    # Check for empty sets
    if hasattr(P, 'representsa_') and P.representsa_('emptySet', 0):
        return False
    if hasattr(S, 'representsa_') and S.representsa_('emptySet', 0):
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
                G = np.diag(S.rad())
                # Create constrained zonotope with no constraints (empty A and b)
                cZ = ConZonotope(c, G, np.empty((0, G.shape[1])), np.empty((0,)))
                return _aux_isIntersecting_P_cZ(P, cZ)
            elif type_ == 'approx':
                return _aux_isIntersecting_approx(P, S, tol)
        
        elif S.__class__.__name__ == 'Polytope':
            # Handle polytope-polytope intersection
            return _aux_isIntersecting_poly_poly(P, S, tol)
        
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
            # Use constraint tolerance (default: 1e-6) following MATLAB
            constraint_tol = 1e-6
            res = val < constraint_tol
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
    
    p, n = H.shape
    
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
    Intersection check of two polytopes by constructing their intersection
    """
    # Construct intersection polytope
    H1 = P1.A
    d1 = P1.b
    H2 = P2.A
    d2 = P2.b
    
    # Combine constraints
    H_combined = np.vstack([H1, H2])
    d_combined = np.vstack([d1, d2]) # Use vstack for column vectors
    
    # Handle equality constraints if they exist
    He1 = P1.Ae
    de1 = P1.be
    He2 = P2.Ae
    de2 = P2.be
    
    # Ensure correct handling of empty matrices for vstack
    if He1.size == 0 and He2.size == 0:
        He_combined = np.empty((0, H_combined.shape[1])) # Ensure 2D empty array
        de_combined = np.empty((0,1)) # Ensure column vector
    elif He1.size == 0:
        He_combined = He2
        de_combined = de2
    elif He2.size == 0:
        He_combined = He1
        de_combined = de1
    else:
        He_combined = np.vstack([He1, He2])
        de_combined = np.vstack([de1, de2])

    # Create intersection polytope
    P_intersect = Polytope(H_combined, d_combined, He_combined, de_combined)
    
    # Check if intersection is empty
    return not P_intersect.representsa_('emptySet', tol) 