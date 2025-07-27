"""
minkDiff - computes the Minkowski difference of two zonotopes.
       A - B = C <-> B + C \subseteq A

Syntax:
    Z = minkDiff(minuend,subtrahend)
    Z = minkDiff(minuend,subtrahend,method)

Inputs:
    minuend - zonotope object
    subtrahend - zonotope object or numerical vector
    method - (optional) used algorithm
     - 'approx' (default)
     - 'exact' (only for 2d or aligned)
     - 'inner'
     - 'outer'
     - 'outer:coarse'
     - 'outer:scaling' (subtrahend must be interval)
     - 'inner:conZonotope'
     - 'inner:RaghuramanKoeln' (implements [2])

Outputs:
    Z - zonotope after Minkowski difference

Example:
    Z1 = zonotope([1 2 2; 0 0 2])
    Z2 = zonotope([0 0.5 0.5 0.3; 1 0 0.5 0.2])

    Z3 = minkDiff(Z1,Z2)
    Z4 = Z2 + Z3

    figure; hold on;
    plot(Z1,[1 2], 'b')
    plot(Z2,[1 2], 'r')
    plot(Z3,[1 2], 'g')
    plot(Z4,[1 2], 'k')

References:
    [1] M. Althoff, "On Computing the Minkowski Difference of Zonotopes",
        arXiv, 2015.
    [2] V. Raghuraman and J. P. Koeln. Set operations and order reductions
        for constrained zonotopes. Automatica, 139, 2022. article no. 110204.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes, conZonotope/minkDiff

Authors:       Matthias Althoff, Niklas Kochdumper, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       10-June-2015 (MATLAB)
Last update:   25-May-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.g.functions.helper.sets.contSet.zonotope.aux_tightenHalfspaces import aux_tightenHalfspaces
from .zonotope import Zonotope
from ..polytope.polytope import Polytope
from ..interval.interval import Interval


def minkDiff(minuend: Zonotope, subtrahend: Union[Zonotope, np.ndarray],
             method: str = 'approx') -> Zonotope:
    """
    Computes the Minkowski difference of two zonotopes.

    Args:
        minuend: Zonotope object (minuend)
        subtrahend: Zonotope object or numerical vector (subtrahend)
        method: Algorithm to use (default: 'approx')

    Returns:
        Zonotope: Result of Minkowski difference

    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # List implemented algorithms
    implemented_algs = ['exact', 'inner', 'outer', 'outer:coarse', 'outer:scaling',
                       'approx', 'inner:conZonotope', 'inner:RaghuramanKoeln']

    # Check inputs
    if not isinstance(minuend, Zonotope):
        raise CORAerror('CORA:wrongValue', 'first', 'zonotope')

    if not isinstance(subtrahend, (Zonotope, np.ndarray)) and not np.isscalar(subtrahend):
        raise CORAerror('CORA:wrongValue', 'second', 'contSet or numeric')

    if method not in implemented_algs:
        raise CORAerror('CORA:wrongValue', 'third', f'Unknown method: {method}')

    # Check if subtrahend is numeric
    if isinstance(subtrahend, (np.ndarray, (int, float))) or np.isscalar(subtrahend):
        return minuend - subtrahend

    # Check if dimensions match
    n = minuend.dim()
    if n != subtrahend.dim():
        raise CORAerror('CORA:dimensionMismatch', minuend, subtrahend)

    # Check if subtrahend is zonotope
    if not isinstance(subtrahend, Zonotope):
        if not (method == 'outer:scaling' and isinstance(subtrahend, Interval)):
            print('CORA:contSet', 'zonotope/minkDiff: Subtrahend is not a zonotope. Enclosing it with a zonotope.')
        # Enclose second set with zonotope
        subtrahend = Zonotope(subtrahend)

    # Check if subtrahend is point
    if subtrahend.G.size == 0:
        return minuend - subtrahend.center

    # Check whether minuend is full dimensional
    if minuend.isFullDim():
        # Solution is exact for n==2 and enforced for this dimension [1,Prop.6]
        if n == 2:
            method = 'exact'

        # Compute Minkowski difference with the approach from [1]
        if method == 'exact':
            if aux_areAligned(minuend, subtrahend):
                # Exact solution for aligned sets according to [1,Prop.5]
                return Zonotope(minuend.c - subtrahend.c,
                              minuend.G - subtrahend.G)
            elif n == 2:
                # Same method as 'inner' [1,Prop.6]
                return aux_minkDiffZono(minuend, subtrahend, method)
            else:
                raise CORAerror('CORA:wrongValue', 'third',
                               'No exact algorithm found: Sets have to be 2-dimensional or aligned.')

        elif method in ['outer', 'outer:coarse', 'inner', 'approx']:
            return aux_minkDiffZono(minuend, subtrahend, method)

        elif method == 'inner:conZonotope':
            # Compute Minkowski difference using constrained zonotopes
            return aux_minkDiffConZono(minuend, subtrahend)

        elif method == 'inner:RaghuramanKoeln':
            # Compute Minkowski difference using [2]
            return aux_RaghuramanKoeln(minuend, subtrahend)

        elif method == 'outer:scaling':
            # Compute Minkowski difference using scaling
            return aux_minkDiffOuterInterval(minuend, subtrahend)
    
    else:
        if subtrahend.isFullDim():
            # Minkowski difference of degenerate minuend and full-dimensional
            # subtrahend is the empty set
            return Zonotope.empty(n)

        if minuend.rank() == subtrahend.rank():
            # Transform the minuend and subtrahend into a space where the
            # minuend is full-dimensional using the singular value decomposition

            # Range of minuend
            U, S, _ = np.linalg.svd(minuend.G)
            new_dim = np.sum(~np.all(withinTol(S, 0)))  # nr. of new dimensions
            P_minuend = U[:new_dim, :]  # projection matrix

            # Range of subtrahend
            U, S, _ = np.linalg.svd(subtrahend.G)
            new_dim = np.sum(~np.all(withinTol(S, 0)))  # nr. of new dimensions
            P_subtrahend = U[:new_dim, :]  # projection matrix

            # Is the range of the minuend and subtrahend equal?
            if (P_minuend.shape == P_subtrahend.shape and
                np.linalg.norm(P_minuend - P_subtrahend) <= 1e-10):
                # Project
                minuend_proj = P_minuend @ minuend  # transformed minuend
                subtrahend_proj = P_minuend @ subtrahend  # transformed subtrahend

                # Solve problem in the transformed domain
                Z_proj = minkDiff(minuend_proj, subtrahend_proj, method)

                # Transform solution back into the original domain
                return np.linalg.pinv(P_minuend) @ Z_proj
            else:
                # No solution exists
                raise CORAerror('CORA:wrongValue', 'first/second',
                               'for non full-dimensional zonotopes: projection matrix found by svd has to be equal')
        else:
            # No solution exists
            raise CORAerror('CORA:wrongValue', 'first/second',
                           'for non full-dimensional zonotopes: rank of generator matrix must be equal')


# Auxiliary functions -----------------------------------------------------

def aux_minkDiffZono(minuend: Zonotope, subtrahend: Zonotope, method: str) -> Zonotope:
    """Compute Minkowski difference using the approach in [1]"""
    
    # Determine generators to be kept
    # Obtain halfspace representation
    P = Polytope(minuend)
    H_orig_twice = P.A
    K_orig_twice = P.b
    H_orig = H_orig_twice[:H_orig_twice.shape[0]//2, :]
    
    # nr of subtrahend generators
    subtrahend_gens = subtrahend.G.shape[1]
    
    # Intersect polytopes according to Theorem 3 of [1]
    delta_K = H_orig_twice @ subtrahend.c
    for i in range(subtrahend_gens):
        delta_K = delta_K + np.abs(H_orig_twice @ subtrahend.G[:, i:i+1])
    K_orig_new = K_orig_twice - delta_K
    
    C = H_orig
    d = K_orig_new[:K_orig_new.shape[0]//2, :]
    
    # Compute center
    c = minuend.c - subtrahend.c
    
    # Obtain minuend generators
    G = minuend.G
    
    # Reverse computation from halfspace generation
    n = minuend.dim()
    if method == 'inner' or (method == 'exact' and n == 2):
        delta_d = d - C @ minuend.c + C @ subtrahend.c
        A_abs = np.abs(C @ G)
        dims = A_abs.shape[1]
        # Vector of cost function
        f = np.linalg.norm(minuend.G, 2, axis=0)
        # A_abs x <= delta_d && x >= 0
        problem = {
            'f': -f,
            'Aineq': np.vstack([A_abs, -np.eye(dims)]),
            'bineq': np.vstack([delta_d, np.zeros((dims, 1))]),
            'Aeq': None,
            'beq': None,
            'lb': None,
            'ub': None
        }
        alpha, _, exitflag = CORAlinprog(problem)[:3]
        if alpha is None or exitflag != 1:
            return Zonotope.empty(n)
        
    elif method in ['outer', 'outer:coarse']:
        # Reduce delta_d using linear programming
        if method == 'outer':
            d_shortened = aux_tightenHalfspaces(H_orig_twice, K_orig_new)
        else:
            d_shortened = K_orig_new
        
        # Is set empty?
        if d_shortened is None or d_shortened.size == 0:
            # Return empty set with correct dimensions
            return Zonotope.empty(n)
        else:
            # Vector of cost function
            f = np.linalg.norm(minuend.G, 2, axis=0)
            # Obtain unrestricted A_abs and delta_d
            C = H_orig
            d = d_shortened[:d_shortened.shape[0]//2, :]
            delta_d = d - C @ minuend.c + C @ subtrahend.c
            A_abs = np.abs(C @ G)
            dims = A_abs.shape[1]
            # A_abs x >= delta_d && x >= 0
            problem = {
                'f': f,
                'Aineq': np.vstack([-A_abs, -np.eye(dims)]),
                'bineq': np.vstack([-delta_d, np.zeros((dims, 1))]),
                'Aeq': None,
                'beq': None,
                'lb': None,
                'ub': None
            }
            alpha, _, exitflag = CORAlinprog(problem)[:3]
    
    elif method == 'approx':
        delta_d = d - C @ minuend.c + C @ subtrahend.c
        A_abs = np.abs(C @ G)
        # Use pseudoinverse to compute an approximation
        alpha = np.linalg.pinv(A_abs) @ delta_d  # solve linear set of equations using the pseudoinverse
    
    else:
        # Should already be caught before
        raise CORAerror('CORA:specialError', f"Unknown method: '{method}'")
    
    # Instantiate Z
    G_new = minuend.G @ np.diag(alpha.flatten())
    # Remove all zero columns
    G_new = G_new[:, ~np.all(G_new == 0, axis=0)]
    return Zonotope(c, G_new)


def aux_minkDiffOuterInterval(minuend: Zonotope, subtrahend: Interval) -> Zonotope:
    """Compute Minkowski difference using scaling"""
    # Subtrahend must be an interval
    if not subtrahend.representsa_('interval', 1e-8):
        raise CORAerror('CORA:wrongValue', 'second', "interval (using method='outer:scaling')")
    
    # Scale using interval enclosure
    rad_min = minuend.interval().rad()
    rad_sub = subtrahend.rad()
    scale = 1 - rad_sub / rad_min  # relative
    return minuend.enlarge(scale)  # outer


def aux_minkDiffConZono(Z1: Zonotope, Z2: Zonotope) -> Zonotope:
    """Compute Minkowski difference based on constrained zonotopes"""
    # Convert first zonotope to constrained zonotope
    cZ = Z1.conZonotope()
    
    # Compute Minkowski difference according to Theorem 1 in [1]
    c = Z2.c
    G = Z2.G
    
    cZ = cZ + (-c)
    
    for i in range(G.shape[1]):
        cZ = cZ.and_(cZ + G[:, i:i+1], cZ + (-G[:, i:i+1]), 'exact')
    
    # Compute zonotope inner-approximation of the constrained zonotope
    return aux_innerApprox(cZ)


def aux_innerApprox(cZ):
    """Inner-approximate a constrained zonotope with a zonotope"""
    # Compute point satisfying all constraints with pseudo inverse
    p_ = np.linalg.pinv(cZ.A) @ cZ.b
    
    # Compute null-space of constraints
    T = np.linalg.null_space(cZ.A)
    
    # Transform boundary constraints of the factor hypercube
    m = cZ.A.shape[1]
    m_ = T.shape[1]
    
    A = np.vstack([np.eye(m), -np.eye(m)])
    b = np.ones((2*m, 1))
    
    A_ = A @ T
    b_ = b - A @ p_
    
    # Construct constraint matrices for linear program
    A = np.hstack([A_, np.abs(A_ @ np.eye(m_))])
    problem = {
        'Aineq': np.vstack([A, np.zeros((m_, m_ + m_))]),
        'bineq': np.vstack([b_, np.zeros((m_, 1))]),
        'Aeq': None,
        'beq': None,
        'lb': None,
        'ub': None
    }
    
    # Construct objective function of the linear program
    problem['f'] = -np.concatenate([np.zeros(m_), np.sum((cZ.G @ T)**2, axis=0)])
    
    # Solve linear program to get interval inner-approximation of polytope
    x, _, exitflag = CORAlinprog(problem)[:3]
    
    # Check if constrained zonotope is empty
    if x is None or exitflag != 1:
        # Return empty set with correct dimensions
        return Zonotope.empty(cZ.dim())
    
    c = x[:m_]
    r = x[m_:]
    r[r < 0] = 0
    int_val = Interval(c - r, c + r)
    
    # Compute transformation matrices
    off = p_ + T @ int_val.center
    S = T @ np.diag(int_val.rad().flatten())
    
    # Construct final zonotope
    c = cZ.c + cZ.G @ off
    G = cZ.G @ S
    
    return Zonotope(c, G)


def aux_RaghuramanKoeln(Z_m: Zonotope, Z_s: Zonotope) -> Zonotope:
    """Solves the Minkowski difference using the method described in [2, Theorem 7]"""
    # Extract data
    c_m = Z_m.c
    c_s = Z_s.c
    G_m = Z_m.G
    G_s = Z_s.G
    M = np.hstack([G_m, G_s])
    
    # Dimension and nr of generators
    n = len(c_m)
    n_m = G_m.shape[1]  # number of generators of Z_m
    n_s = G_s.shape[1]  # number of generators of Z_s
    
    # Create M_tilde
    M_tilde = np.zeros((n * (n_m + n_s), n_m + n_s))
    for i in range(M.shape[1]):
        M_tilde[n*i:n*(i+1), i] = M[:, i]
    
    # linprog solved linear programs in the form:
    # min_x f^T x
    # such that:
    # Ax <= b
    # A_eq x = b_eq
    # x_l <= x <= x_u
    
    # A
    a = np.kron(np.ones((1, n_m + 2*n_s + 1)), np.eye(n_m))
    I = np.eye(n_m * (n_m + 2 * n_s + 1))
    
    A_ineq = np.vstack([
        np.hstack([np.zeros((n_m, n_m+n_s)), np.zeros((n_m, n_m*(n_m + 2 * n_s + 1))), a, np.zeros((n_m, n))]),
        np.hstack([np.zeros((n_m*(n_m + 2 * n_s + 1), n_m+n_s)), I, -I, np.zeros((n_m*(n_m + 2 * n_s + 1), n))]),
        np.hstack([np.zeros((n_m*(n_m + 2 * n_s + 1), n_m+n_s)), -I, -I, np.zeros((n_m*(n_m + 2 * n_s + 1), n))])
    ])
    
    # b
    b_ineq = np.vstack([
        np.ones((n_m, 1)),
        np.zeros((2*n_m*(n_m + 2 * n_s + 1), 1))
    ])
    
    # A_eq
    A_eq = np.vstack([
        np.hstack([M_tilde, -np.kron(np.eye(n_m+n_s), G_m), np.zeros((n*(n_m + n_s), n_m*n_s)), 
                   np.zeros((n*(n_m + n_s), n_m)), np.zeros((n*(n_m + n_s), n_m*(n_m + 2 * n_s + 1))), np.zeros((n*(n_m + n_s), n))]),
        np.hstack([np.zeros((n*n_s, n_m+n_s)), np.zeros((n*n_s, n_m*(n_m + n_s))), -np.kron(np.eye(n_s), G_m), 
                   np.zeros((n*n_s, n_m)), np.zeros((n*n_s, n_m*(n_m + 2 * n_s + 1))), np.zeros((n*n_s, n))]),
        np.hstack([np.zeros((n, n_m+n_s)), np.zeros((n, n_m*(n_m + n_s))), np.zeros((n, n_m*n_s)), -G_m, 
                   np.zeros((n, n_m*(n_m + 2 * n_s + 1))), -np.eye(n)])
    ])
    
    # b_eq
    b_eq = np.vstack([
        np.zeros((n*(n_m + n_s), 1)),
        -G_s.flatten().reshape(-1, 1),
        (c_s - c_m).reshape(-1, 1)
    ])
    
    # f minimizes phi
    problem = {
        'f': np.concatenate([-np.ones(n_m+n_s), np.zeros(2*n_m*(n_m + 2 * n_s + 1)+n)]),
        'Aineq': A_ineq,
        'bineq': b_ineq,
        'Aeq': A_eq,
        'beq': b_eq,
        'lb': None,
        'ub': None
    }
    
    # Solve linear programming problem
    x, _, exitflag = CORAlinprog(problem)[:3]
    
    if exitflag == 1:
        # Extract phi
        phi = x[:n_m+n_s]
        # Extract c_d
        c_d = x[-n:]
        # Result
        return Zonotope(np.column_stack([c_d, M @ np.diag(phi.flatten())]))
    elif exitflag == -2:
        # No feasible point found -> empty
        return Zonotope.empty(n)
    else:
        # Unknown error
        raise CORAerror("CORA:specialError", 'No solution exists.')


def aux_areAligned(minuend: Zonotope, subtrahend: Zonotope) -> bool:
    """Check if generators are aligned"""
    # Extract generators
    G_min = minuend.G
    G_sub = subtrahend.G
    
    # Check dimensions
    if G_min.shape == G_sub.shape:
        # Normalize
        norm_min = np.max(np.linalg.norm(G_min, axis=0))
        norm_sub = np.max(np.linalg.norm(G_sub, axis=0))
        
        if withinTol(norm_min, 0) and withinTol(norm_sub, 0):
            # Both have only all-zero generators
            return True
        elif withinTol(norm_min, 0) or withinTol(norm_sub, 0):
            # Only one generator matrix has only all-zero generators
            return False
        else:
            # Normalize generators
            G_min = G_min / norm_min
            G_sub = G_sub / norm_sub
            
            # Generators have to be ordered
            return compareMatrices(G_min, G_sub, 1e-8, 'equal', True)
    else:
        return False


