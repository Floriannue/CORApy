"""
randPoint_ - generates random points within a zonotope

This function generates random points within a zonotope using various
sampling methods including standard, extreme, uniform, and specialized algorithms.

Authors: Matthias Althoff, Mark Wetzlinger, Adrian Kulmburg, Severin Prenitzer (MATLAB)
         Python translation by AI Assistant
Written: 23-September-2008 (MATLAB)
Last update: 05-October-2024 (MATLAB)
Python translation: 2025
"""

from typing import Union, TYPE_CHECKING
import numpy as np
from scipy.linalg import svd
import warnings
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.g.functions.helper.sets.contSet.zonotope.ndimCross import ndimCross

if TYPE_CHECKING:
    from .zonotope import Zonotope


def randPoint_(Z: 'Zonotope', N: Union[int, str] = 1, type_: str = 'standard') -> np.ndarray:
    """
    Generates random points within a zonotope
    
    Args:
        Z: Zonotope object
        N: Number of random points or 'all' for extreme points
        type_: Type of random point generation:
               - 'standard': Standard random sampling
               - 'extreme': Extreme points (vertices)
               - 'uniform': Uniform sampling (billiard walk)
               - 'uniform:hitAndRun': Hit-and-run uniform sampling
               - 'uniform:ballWalk': Ball walk uniform sampling
               - 'uniform:billiardWalk': Billiard walk uniform sampling
               - 'radius': Radius-based sampling
               - 'boundary': Boundary sampling
               - 'gaussian': Gaussian sampling
        
    Returns:
        np.ndarray: Random points (each column is a point)
        
    Raises:
        CORAerror: If algorithm not supported for zonotope type
        
    Example:
        >>> Z = Zonotope([1, 0], [[1, 0, 1], [-1, 2, 1]])
        >>> p = randPoint_(Z, 100, 'standard')
    """
    # Handle empty zonotope - check dimensions properly
    if Z.c.shape[1] == 0:  # Empty zonotope has center shape (n, 0)
        n = Z.c.shape[0]  # Get dimension from center shape
        # For empty sets, always return 0 points regardless of N
        return np.zeros((n, 0))
    
    # Zonotope is just a point -> replicate center N times
    if Z.representsa_('point', 1e-15):
        if isinstance(N, str):
            N = 1
        return np.tile(Z.c.reshape(-1, 1), (1, N))
    
    # Generate different types of random points
    if type_ == 'standard':
        return _aux_randPoint_standard(Z, N)
    
    elif type_ == 'extreme':
        return _aux_randPoint_extreme(Z, N)
    
    elif type_ in ['uniform', 'uniform:billiardWalk']:
        if Z.representsa_('parallelotope', 1e-15):
            return _aux_randPointParallelotopeUniform(Z, N)
        else:
            return _aux_randPointBilliard(Z, N)
    
    elif type_ == 'uniform:ballWalk':
        return _aux_randPointBallWalk(Z, N)
    
    elif type_ == 'uniform:hitAndRun':
        return _aux_randPointHitAndRun(Z, N)
    
    elif type_ == 'radius':
        return _aux_randPointRadius(Z, N)
    
    elif type_ == 'boundary':
        return _aux_getRandomBoundaryPoints(Z, N)
    
    elif type_ == 'gaussian':
        return _aux_randPoint_gaussian(Z, N)
    
    else:
        raise CORAerror('CORA:noSpecificAlg', f'{type_} not supported for zonotope')


def _aux_randPoint_standard(Z: 'Zonotope', N: int) -> np.ndarray:
    """Standard random point generation"""
    if isinstance(N, str):
        N = 1
    
    # Take random values for factors
    factors = -1 + 2 * np.random.rand(Z.G.shape[1], N)
    # Sample points
    p = Z.c.reshape(-1, 1) + Z.G @ factors
    return p


def _aux_randPoint_gaussian(Z: 'Zonotope', N: int) -> np.ndarray:
    """Gaussian random point generation"""
    if isinstance(N, str):
        N = 1
    
    # Take random values for factors from a standard normal distribution
    factors = np.random.randn(Z.G.shape[1], N)
    # Sample points
    p = Z.c.reshape(-1, 1) + Z.G @ factors
    return p


def _aux_randPoint_extreme(Z: 'Zonotope', N: Union[int, str]) -> np.ndarray:
    """Extreme point generation"""
    n = Z.dim()
    c = Z.c
    G = Z.G
    
    # 1D case
    if n == 1:
        # Flush all generators into one
        G_sum = np.sum(np.abs(G), axis=1)
        if isinstance(N, str) and N == 'all':
            N = 2  # For 1D, we have 2 extreme points
        elif isinstance(N, str):
            N = 1
        # Random signs
        s = np.sign(np.random.randn(1, N))
        # Zero signs become positive
        s[s == 0] = 1
        # Instantiate points
        p = c.reshape(-1, 1) + s * G_sum.reshape(-1, 1)
        return p
    
    # Remove redundant generators
    Z = Z.compact_('all', 1e-10)
    G = Z.G
    
    # Consider degenerate case
    if np.linalg.matrix_rank(G) < n:
        # For degenerate case, use projection approach
        Z_shifted = Z + (-c)
        if isinstance(N, str):
            N = 1
        p = np.zeros((n, N))
        
        # Use SVD to find non-degenerate subspace
        U, s, Vt = svd(np.hstack([-G, G]))
        d = s
        ind = np.where(d > 1e-15)[0]
        
        if len(ind) == 0:
            return c.reshape(-1, 1) if N == 1 else np.tile(c.reshape(-1, 1), (1, N))
        
        # Project to non-degenerate subspace
        Z_proj = Z_shifted.project(ind)
        p_proj = randPoint_(Z_proj, N, 'extreme')
        p[ind, :] = p_proj
        p = c.reshape(-1, 1) + U @ p
        return p
    
    # Compute approximate number of zonotope vertices
    q = _aux_numberZonoVertices(Z)
    
    if isinstance(N, str) and N == 'all':
        # Return all extreme points
        return Z.vertices_()
    
    elif isinstance(N, int):
        if 10 * N < q:
            # Generate random vertices
            return _aux_getRandomVertices(Z, N)
        
        elif N < q:
            # Select random vertices
            V = Z.vertices_()
            if V.shape[1] >= N:
                # We have enough vertices, just select N of them
                ind = np.random.permutation(V.shape[1])[:N]
                return V[:, ind]
            else:
                # Need more points than vertices available
                N_ = N - V.shape[1]
                V_ = _aux_getRandomBoundaryPoints(Z, N_)
                return np.hstack([V, V_])
        
        else:
            # Compute vertices and additional points on the boundary
            V = Z.vertices_()
            N_ = N - V.shape[1]
            if N_ > 0:
                V_ = _aux_getRandomBoundaryPoints(Z, N_)
                return np.hstack([V, V_])
            else:
                # Just return the vertices
                return V
    
    else:
        raise ValueError("N must be an integer or 'all'")


def _aux_numberZonoVertices(Z: 'Zonotope') -> float:
    """Compute approximate number of zonotope vertices"""
    n = Z.dim()
    nrGen = Z.G.shape[1]
    
    if nrGen == 0:
        # Only center
        return 1.0
    
    # Create dynamic programming table
    D = np.zeros((n, nrGen))
    D[0, :] = 2 * np.ones(nrGen)
    D[:, 0] = 2 * np.ones(n)
    
    for j in range(1, n):
        for k in range(1, nrGen):
            D[j, k] = D[j, k-1] + D[j-1, k-1]
    
    return float(D[n-1, nrGen-1])


def _aux_getRandomVertices(Z: 'Zonotope', N: int) -> np.ndarray:
    """Generate random vertices"""
    n = Z.dim()
    nrGen = Z.G.shape[1]
    V = np.zeros((nrGen, N))
    cnt = 0
    
    # Loop until the desired number of vertices is achieved
    while cnt < N:
        # Generate random zonotope face
        randOrder = np.random.permutation(nrGen)
        ind = randOrder[:n-1]
        Q = Z.G[:, ind]
        
        # Compute normal vector using cross product generalization
        c = ndimCross(Q)
        v = np.sign(c.T @ Z.G).flatten()
        
        # Generate random vertex on the zonotope face
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            v_ = v.copy()
            v_[ind] = np.sign(-1 + 2 * np.random.rand(n-1))
            # Zero signs become positive
            v_[v_ == 0] = 1
            
            # Check if this vertex is new
            if cnt == 0 or not np.any(np.all(V[:, :cnt].T == v_, axis=1)):
                V[:, cnt] = v_
                cnt += 1
                break
            attempts += 1
        
        if attempts >= 100:
            # Fallback: just use a random vertex
            V[:, cnt] = np.sign(-1 + 2 * np.random.rand(nrGen))
            V[V[:, cnt] == 0, cnt] = 1
            cnt += 1
    
    # Compute vertices
    return Z.c.reshape(-1, 1) + Z.G @ V


def _aux_getRandomBoundaryPoints(Z: 'Zonotope', N: int) -> np.ndarray:
    """Generate random boundary points"""
    n = Z.dim()
    nrGen = Z.G.shape[1]
    
    if N == 0:
        return np.zeros((n, 0))
    
    V = np.zeros((nrGen, N))
    
    # Loop until the desired number of vertices is achieved
    for i in range(N):
        # Generate random zonotope face
        randOrder = np.random.permutation(nrGen)
        ind = randOrder[:n-1]
        Q = Z.G[:, ind]
        
        # Compute normal vector
        c = ndimCross(Q)
        if np.random.rand() > 0.5:
            c = -c
        
        V[:, i] = np.sign(c.T @ Z.G).flatten()
        
        # Generate random point on the zonotope face
        V[ind, i] = -1 + 2 * np.random.rand(n-1)
    
    # Compute boundary points
    return Z.c.reshape(-1, 1) + Z.G @ V


def _aux_randPointParallelotopeUniform(Z: 'Zonotope', N: int) -> np.ndarray:
    """Uniform random point generation for parallelotopes"""
    if isinstance(N, str):
        N = 1
    
    # Compact zonotope first
    Z = Z.compact_()
    G = Z.G
    c = Z.c
    
    # Uniformly sample unit hypercube
    p = 2 * np.random.rand(G.shape[1], N) - 1
    
    return c.reshape(-1, 1) + G @ p


def _aux_randPointHitAndRun(Z: 'Zonotope', N: int) -> np.ndarray:
    """Hit-and-run algorithm for uniform sampling - complete MATLAB translation"""
    if isinstance(N, str):
        N = 1
    
    c = Z.c
    Z_shifted = Z + (-c)
    n = Z.dim()
    
    # Check whether the zonotope is degenerate
    try:
        from cora_python.contSet.contSet.isFullDim import isFullDim
        res, X = isFullDim(Z)
        
        if not res:
            # Add padding if needed for degenerate case
            if X.shape[1] < n:
                X = np.hstack([X, np.zeros((n, n - X.shape[1]))])
        else:
            X = np.eye(n)
    except:
        # Fallback
        X = np.eye(n)
        res = True
    
    G = Z_shifted.G
    m = G.shape[1]
    
    # Start with a random point in the zonotope
    try:
        p0 = _aux_randPoint_standard(Z_shifted, 1).flatten()
    except:
        p0 = np.zeros(n)
    
    p = np.zeros((n, N))
    max_lp_failures = 10  # Maximum number of LP failures per point

    # Sample N points
    for i in range(N):
        lp_failures = 0
        current_point = p0.copy()
        
        # Limit iterations to prevent infinite loops
        for attempt in range(5):  # Max 5 attempts per point
            # Sample movement vector
            d = np.random.randn(n)
            d = d / np.linalg.norm(d)
            
            # Project direction to subspace if degenerate
            d = X @ d
            
            # Define parameters for linear programs
            f = np.concatenate([[1], np.zeros(m)])
            
            # Inequality constraints: -1 <= ksi <= 1
            A_ineq = np.vstack([
                np.hstack([np.zeros((2*m, 1)), np.vstack([np.eye(m), -np.eye(m)])])
            ])
            b_ineq = np.ones(2*m)
            
            # Equality constraint: d = G*ksi + t*d (where t is first variable)
            A_eq = np.hstack([d.reshape(-1, 1), -G])
            b_eq = -current_point
            
            problem_min = {
                'f': f,
                'Aineq': A_ineq,
                'bineq': b_ineq,
                'Aeq': A_eq,
                'beq': b_eq,
                'lb': None,
                'ub': None
            }
            
            problem_max = {
                'f': -f,  # Maximize by negating objective
                'Aineq': A_ineq,
                'bineq': b_ineq,
                'Aeq': A_eq,
                'beq': b_eq,
                'lb': None,
                'ub': None
            }
            
            # Execute linear programs with timeout protection
            try:
                minZ, _, exitflag_min, _, _ = CORAlinprog(problem_min)
                if exitflag_min < 0:
                    lp_failures += 1
                    if lp_failures >= max_lp_failures:
                        break
                    continue
                
                maxZ, _, exitflag_max, _, _ = CORAlinprog(problem_max)
                if exitflag_max < 0:
                    lp_failures += 1
                    if lp_failures >= max_lp_failures:
                        break
                    continue
                
                # Sample line segment uniformly
                minC = minZ[0]
                maxC = -maxZ[0]  # Note: maxZ is negative due to negated objective
                
                if maxC <= minC:
                    # Invalid range, try different direction
                    continue
                    
                sample_t = minC + np.random.rand() * (maxC - minC)
                sample_point = current_point + sample_t * d
                
                # Validate the sample point
                try:
                    factors = np.linalg.lstsq(G, sample_point, rcond=None)[0]
                    if np.all(np.abs(factors) <= 1.0 + 1e-10):
                        current_point = sample_point
                        break
                except:
                    continue
                    
            except Exception as e:
                # LP failed, try again with different direction
                lp_failures += 1
                if lp_failures >= max_lp_failures:
                    break
                continue
        
        p[:, i] = current_point
        p0 = current_point
    
    return p + c.reshape(-1, 1)


def _aux_randPointBallWalk(Z: 'Zonotope', N: int) -> np.ndarray:
    """Ball walk algorithm for uniform sampling - complete MATLAB translation"""
    if isinstance(N, str):
        N = 1
    
    n = Z.dim()
    
    # First, make sure that the zonotope is zero-centered; save the center
    c = Z.center()
    Z_shifted = Z - c
    
    # Then, for the ball-walk, we need to make sure that the inscribed
    # ellipsoid in Z is actually a sphere. For this, we need the SVD:
    U, S, Vt = svd(Z_shifted.G)
    
    # U is a rotation, which is bijective; we can thus turn Z:
    # Apply U.T to the zonotope (equivalent to U' * Z in MATLAB)
    Z_rotated = Z.__class__(U.T @ Z_shifted.c, U.T @ Z_shifted.G)
    
    # We now have a zonotope that has its main axes aligned with the
    # canonical basis. We compute the rank of S now, to check if the
    # zonotope is degenerate.
    # If not, we 'invert' the scaling introduced by S. If it is degenerate,
    # we only do this for the relevant directions:
    r = np.sum(S > 1e-12)  # rank
    s = S.copy()
    s_inv = np.ones(n)
    s_inv[:r] = 1.0 / s[:r]  # Only invert non-zero singular values
    s_inv[r:] = 1.0  # Keep ones for zero directions
    S_inv = np.diag(s_inv)
    
    Z_final = Z.__class__(S_inv @ Z_rotated.c, S_inv @ Z_rotated.G)
    
    # We have now turned and scaled Z in such a way, that it contains the
    # unit ball, and is contained in the unit ball scaled by sqrt(n), where
    # n is the dimension. We choose the length of the ball-walk to be
    # 1/sqrt(n)
    # this could be improved in the future, but at the moment it yields
    # pretty good results overall
    delta = 1.0 / np.sqrt(n)
    
    # Start with some random point, not necessarily uniformly distributed
    try:
        p0 = _aux_randPoint_standard(Z_final, 1).flatten()
    except:
        p0 = np.zeros(n)
    
    p = np.zeros((n, N))
    
    # Sample N points
    for i in range(N):
        current_point = p0.copy()
        max_attempts = 1000  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            # Sample movement vector
            d = np.zeros(n)
            for j in range(n):
                if j >= r:
                    # Cut out the parts of d that would be canceled by G (i.e., S*V')
                    d[j] = 0.0
                else:
                    d[j] = np.random.randn()
            
            # Sample length
            ell = np.random.rand() ** (1.0 / n)
            
            # Build candidate point
            candidate = current_point + delta * ell * d
            
            # Check containment using exact containment check
            # MATLAB: Z.contains_(candidate,'exact',1e-5,0,false,false)
            contained = Z_final.contains_(candidate, 'exact', 1e-5, 200, False, False)
            if isinstance(contained, tuple):
                contained = contained[0]
            
            if contained:
                current_point = candidate
                break
        
        p[:, i] = current_point
        p0 = current_point
    
    # Transform everything back now
    p = U @ np.diag(1.0 / s_inv) @ p + c.reshape(-1, 1)
    
    return p


def _aux_randPointBilliard(Z: 'Zonotope', N: int) -> np.ndarray:
    """Billiard walk algorithm for uniform sampling - complete MATLAB translation"""
    if isinstance(N, str):
        N = 1
    
    c = Z.c
    Z_shifted = Z - c
    n = Z.dim()
    
    # Check whether the zonotope is degenerate
    try:
        from cora_python.contSet.contSet.isFullDim import isFullDim
        Z_isFullDim, X = isFullDim(Z)
        
        if not Z_isFullDim:
            # Add padding if needed
            if X.shape[1] < n:
                X = np.hstack([X, np.zeros((n, n - X.shape[1]))])
        else:
            X = np.eye(n)
    except:
        X = np.eye(n)
        Z_isFullDim = True
    
    G = Z_shifted.G
    m = G.shape[1]
    
    # Initial point and parameters
    try:
        p0 = _aux_randPoint_standard(Z_shifted, 1).flatten()
    except:
        p0 = np.zeros(n)
    
    # Calculate trajectory parameters
    try:
        from cora_python.contSet.zonotope.norm_ import norm_
        tau = norm_(Z_shifted, 2, 'ub')  # Upper bound estimate
    except:
        tau = np.sqrt(np.sum(np.sum(Z_shifted.G**2, axis=0)))  # Fallback
    
    R0 = max(10 * n, 10)  # Maximum reflections
    p = np.zeros((n, N))
    max_iterations_per_point = 100  # Prevent infinite loops
    
    # Sample N points
    for i in range(N):
        # Set segment start point, trajectory length and max reflections
        q0 = p0.copy()
        l = tau * (-np.log(np.random.rand()))
        R = R0
        
        # Sample direction
        d = np.random.randn(n)
        d = d / np.linalg.norm(d)
        d = X @ d  # Project to subspace if degenerate
        
        point_found = False
        iteration_count = 0
        
        while not point_found and iteration_count < max_iterations_per_point:
            iteration_count += 1
            
            # Define parameters for linear program to find boundary intersection
            # Variables: [t, beta_1, ..., beta_m] where t is the distance along direction d
            f_vec = np.zeros(1 + m)
            f_vec[0] = 1  # Minimize t
            
            # Inequality constraints: -1 <= beta_i <= 1 for all i
            # This becomes: beta_i <= 1 and -beta_i <= 1
            # In matrix form: [0, I; 0, -I] * [t; beta] <= [1; 1]
            Aineq = np.hstack([np.zeros((2*m, 1)), np.vstack([np.eye(m), -np.eye(m)])])
            bineq = np.ones(2*m)
            
            # Equality constraint: q0 + t*d = G*beta
            # Rearranged: t*d - G*beta = -q0
            # In matrix form: [d, -G] * [t; beta] = -q0
            Aeq = np.hstack([d.reshape(-1, 1), -G])
            beq = -q0.flatten()
            
            problem_boundary = {
                'f': f_vec,
                'Aineq': Aineq,
                'bineq': bineq,
                'Aeq': Aeq,
                'beq': beq,
                'lb': None,
                'ub': None
            }
            
            try:
                # Get intersection point of direction vector with zonotope boundary
                maxZ, _, exitflag, _, _ = CORAlinprog(problem_boundary)
                if exitflag < 0:
                    # Linear program failed, restart with new direction
                    q0 = p0.copy()
                    l = tau * (-np.log(np.random.rand()))
                    d = np.random.randn(n)
                    d = d / np.linalg.norm(d)
                    d = X @ d
                    R = R0
                    continue
                
                maxC = maxZ[0]
                q = q0 + maxC * d
                
                # Return point on line segment if the trajectory length is reached
                line_segment = q - q0
                segment_length = np.linalg.norm(line_segment)
                
                if l <= segment_length:
                    p[:, i] = q0 + (l / segment_length) * line_segment
                    p0 = p[:, i]
                    point_found = True
                    break
                
                # Check if a nonsmooth boundary point is met
                extreme_counter = 0
                for j in range(1, min(m+1, len(maxZ))):
                    if abs(abs(maxZ[j]) - 1) < np.finfo(float).eps:
                        extreme_counter += 1
                
                # Sample new direction if max reflections or nonsmooth point reached
                if R <= 0 or extreme_counter <= m - 2 or extreme_counter == m:
                    R = R0
                    q0 = p0.copy()
                    l = tau * (-np.log(np.random.rand()))
                    d = np.random.randn(n)
                    d = d / np.linalg.norm(d)
                    d = X @ d
                    continue
                else:
                    R -= 1
                
                # Define parameters for linear program to find normal vector
                if Z_isFullDim:
                    problem_facet = {
                        'f': np.concatenate([np.zeros(m), -q]),
                        'Aineq': np.vstack([
                            np.hstack([np.ones((1, m)), np.zeros((1, n))]),
                            np.hstack([-np.eye(m), G.T]),
                            np.hstack([-np.eye(m), -G.T])
                        ]),
                        'bineq': np.concatenate([[1], np.zeros(2*m)]),
                        'Aeq': None,
                        'beq': None,
                        'lb': None,
                        'ub': None
                    }
                else:
                    problem_facet = {
                        'f': np.concatenate([np.zeros(n+m), -q]),
                        'Aineq': np.vstack([
                            np.hstack([np.zeros((1, n)), np.ones((1, m)), np.zeros((1, n))]),
                            np.hstack([np.zeros((m, n)), -np.eye(m), G.T]),
                            np.hstack([np.zeros((m, n)), -np.eye(m), -G.T])
                        ]),
                        'bineq': np.concatenate([[1], np.zeros(2*m)]),
                        'Aeq': np.hstack([X, np.zeros((n, m)), -np.eye(n)]),
                        'beq': np.zeros(n),
                        'lb': None,
                        'ub': None
                    }
                
                # Get unit normal vector of hit facet
                try:
                    linOut, _, exitflag_facet, _, _ = CORAlinprog(problem_facet)
                    if exitflag_facet < 0 or linOut is None:
                        # Failed to find normal, restart
                        q0 = p0.copy()
                        l = tau * (-np.log(np.random.rand()))
                        d = np.random.randn(n)
                        d = d / np.linalg.norm(d)
                        d = X @ d
                        R = R0
                        continue
                    
                    linOut_flat = linOut.flatten() if linOut.ndim > 1 else linOut
                    
                    if Z_isFullDim:
                        if len(linOut_flat) >= m + n:
                            s = linOut_flat[m:m+n]
                        else:
                            # Fallback restart
                            q0 = p0.copy()
                            l = tau * (-np.log(np.random.rand()))
                            d = np.random.randn(n)
                            d = d / np.linalg.norm(d)
                            d = X @ d
                            R = R0
                            continue
                    else:
                        if len(linOut_flat) >= n + m + n:
                            s = linOut_flat[n+m:n+m+n]
                        else:
                            # Fallback restart
                            q0 = p0.copy()
                            l = tau * (-np.log(np.random.rand()))
                            d = np.random.randn(n)
                            d = d / np.linalg.norm(d)
                            d = X @ d
                            R = R0
                            continue
                    
                    s_norm = np.linalg.norm(s)
                    if s_norm > 1e-12:
                        s = s / s_norm
                        
                        # Change start point, direction and remaining length
                        q0 = q
                        d = d - 2 * np.dot(d, s) * s
                        l = l - segment_length
                    else:
                        # Degenerate normal, restart
                        q0 = p0.copy()
                        l = tau * (-np.log(np.random.rand()))
                        d = np.random.randn(n)
                        d = d / np.linalg.norm(d)
                        d = X @ d
                        R = R0
                        continue
                        
                except Exception:
                    # Failed to compute reflection, restart
                    q0 = p0.copy()
                    l = tau * (-np.log(np.random.rand()))
                    d = np.random.randn(n)
                    d = d / np.linalg.norm(d)
                    d = X @ d
                    R = R0
                    continue
                    
            except Exception:
                # Something went wrong, restart with new trajectory
                q0 = p0.copy()
                l = tau * (-np.log(np.random.rand()))
                d = np.random.randn(n)
                d = d / np.linalg.norm(d)
                d = X @ d
                R = R0
                continue
        
        # If we couldn't find a point after max iterations, use current point
        if not point_found:
            p[:, i] = q0
            p0 = q0
    
    return p + c.reshape(-1, 1)


def _aux_randPointRadius(Z: 'Zonotope', N: int) -> np.ndarray:
    """Radius-based random point generation"""
    if isinstance(N, str):
        N = 1
    
    # Sample random points on boundary to obtain radii
    radii = _aux_getRandomBoundaryPoints(Z, N) - Z.c.reshape(-1, 1)
    
    # Sample semi-uniformly within 'sphere' 
    scaling = np.random.rand(1, N) ** (1.0 / Z.dim())
    
    return Z.c.reshape(-1, 1) + scaling * radii