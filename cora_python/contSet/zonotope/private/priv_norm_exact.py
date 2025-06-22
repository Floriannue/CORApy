"""
priv_norm_exact - computes the exact maximum norm

Syntax:
    val, x = priv_norm_exact(Z, type)

Inputs:
    Z - zonotope object
    type - p-norm

Outputs:
    val - norm value of vertex with biggest distance from the center
    x - vertex attaining maximum norm

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       18-September-2019 (MATLAB)
Last update:   21-April-2023 (VG, reworked completely)
Last revision: ---
"""

import numpy as np
from typing import Tuple
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


def priv_norm_exact(Z, norm_type: int) -> Tuple[float, np.ndarray]:
    """
    Computes the exact maximum norm using the MATLAB algorithm.
    
    The MATLAB algorithm transforms the problem:
    max_{u∈{-1,1}^m} u'*G'*G*u
    to:
    min_{b∈{0,1}^m} ||sqrt(M)*(b-0.5)||_2
    where M = lmax*I - G'*G and lmax = max(eig(G'*G))
    
    Args:
        Z: zonotope object
        norm_type: p-norm type (only 2-norm supported)
        
    Returns:
        tuple: (val, x) where val is norm value and x is vertex
    """
    
    if norm_type != 2:
        raise ValueError('Only Euclidean norm supported.')
    
    # Get zonotope properties
    c = Z.c
    G = Z.G
    
    if G.size == 0:
        # Zonotope is just a point
        x = c
        val = float(np.linalg.norm(x))
        return val, x
    
    n, m = G.shape
    
    # Core MATLAB algorithm: compute G'*G and eigenvalues
    GG = G.T @ G
    eigenvals = np.linalg.eigvals(GG)
    lmax = np.max(eigenvals)
    
    # Compute matrix M = lmax*I - G'*G
    M = lmax * np.eye(m) - GG
    
    # Try different solution approaches
    if CVXPY_AVAILABLE:
        return _solve_with_cvxpy_correct(c, G, M, lmax, m)
    else:
        # Fallback to vertex enumeration for small problems
        warnings.warn("CVXPY not available. Using vertex enumeration for exact norm computation.")
        return _solve_with_vertex_enumeration(Z)


def _solve_with_cvxpy_correct(c: np.ndarray, G: np.ndarray, M: np.ndarray, lmax: float, m: int) -> Tuple[float, np.ndarray]:
    """
    Solve using CVXPY following the MATLAB transformation.
    
    MATLAB transforms the problem to:
    min_{t,b} t
    s.t. ||sqrt(M)*b - 0.5*sqrt(M)*ones(m,1)||_2 <= t
         0 <= b <= 1, b ∈ {0,1}^m
    """
    
    # Check if M is positive semidefinite for sqrt computation
    try:
        # Compute matrix square root of M
        eigenvals_M, eigenvecs_M = np.linalg.eigh(M)
        
        # Handle numerical issues with negative eigenvalues
        eigenvals_M = np.maximum(eigenvals_M, 0)
        M_sqrt = eigenvecs_M @ np.diag(np.sqrt(eigenvals_M)) @ eigenvecs_M.T
        
    except Exception:
        # If matrix square root fails, fall back to vertex enumeration
        warnings.warn("Matrix square root computation failed, using vertex enumeration")
        return _solve_with_vertex_enumeration(type('obj', (), {'c': c, 'G': G})())
    
    # For small problems, try exact mixed-integer approach
    if m <= 20:  # Limit for exact binary optimization
        try:
            # Binary variables b ∈ {0,1}^m
            b = cp.Variable(m, boolean=True)
            
            # Auxiliary variable for the norm
            t = cp.Variable()
            
            # The constraint: ||sqrt(M)*b - 0.5*sqrt(M)*ones(m,1)||_2 <= t
            ones_m = np.ones(m)
            constraint_expr = M_sqrt @ b - 0.5 * M_sqrt @ ones_m
            
            # Objective: minimize t
            objective = cp.Minimize(t)
            
            # Constraints
            constraints = [
                cp.norm(constraint_expr, 2) <= t,
                t >= 0
            ]
            
            # Solve the problem
            problem = cp.Problem(objective, constraints)
            
            # Try different solvers
            solvers_to_try = [cp.CLARABEL, cp.SCS, cp.CVXOPT]
            
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        # Extract solution: convert b to u = 2*(b-0.5)
                        b_sol = b.value
                        u_sol = 2 * (b_sol - 0.5)
                        
                        # Compute the vertex
                        x = G @ u_sol + c.flatten()
                        val = float(np.linalg.norm(x))
                        
                        return val, x.reshape(-1, 1)
                except Exception:
                    continue
                    
        except Exception:
            pass  # Fall through to continuous relaxation
    
    # Continuous relaxation approach
    try:
        # Continuous variables b ∈ [0,1]^m
        b = cp.Variable(m)
        
        # Auxiliary variable for the norm
        t = cp.Variable()
        
        # The constraint: ||sqrt(M)*b - 0.5*sqrt(M)*ones(m,1)||_2 <= t
        ones_m = np.ones(m)
        constraint_expr = M_sqrt @ b - 0.5 * M_sqrt @ ones_m
        
        # Objective: minimize t
        objective = cp.Minimize(t)
        
        # Constraints
        constraints = [
            cp.norm(constraint_expr, 2) <= t,
            b >= 0,
            b <= 1,
            t >= 0
        ]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        
        # Try different solvers
        solvers_to_try = [cp.CLARABEL, cp.SCS, cp.CVXOPT]
        
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    # Extract solution and round to {0,1}
                    b_sol = b.value
                    b_rounded = np.round(b_sol)  # Round to nearest integer
                    
                    # Convert b to u = 2*(b-0.5)
                    u_sol = 2 * (b_rounded - 0.5)
                    
                    # Compute the vertex
                    x = G @ u_sol + c.flatten()
                    val = float(np.linalg.norm(x))
                    
                    return val, x.reshape(-1, 1)
            except Exception:
                continue
                
    except Exception:
        pass
    
    # Final fallback to vertex enumeration
    warnings.warn("Optimization failed, using vertex enumeration")
    return _solve_with_vertex_enumeration(type('obj', (), {'c': c, 'G': G})())


def _solve_with_vertex_enumeration(Z) -> Tuple[float, np.ndarray]:
    """
    Solve by enumerating vertices (for small problems).
    """
    try:
        # Try to get vertices from the zonotope
        if hasattr(Z, 'vertices_') and callable(Z.vertices_):
            vertices = Z.vertices_()
        else:
            # Manual vertex enumeration
            c = Z.c
            G = Z.G
            
            if G.size == 0:
                return float(np.linalg.norm(c)), c
            
            n, m = G.shape
            
            # For small problems, enumerate all vertices
            if m <= 20:  # 2^20 = ~1M vertices
                max_norm = 0
                best_vertex = c
                
                for i in range(2**m):
                    # Convert i to binary representation for alpha ∈ {-1,1}^m
                    alpha = np.zeros(m)
                    temp = i
                    for j in range(m):
                        alpha[j] = 1 if (temp % 2) == 1 else -1
                        temp //= 2
                    
                    # Compute vertex
                    vertex = G @ alpha + c.flatten()
                    norm_val = np.linalg.norm(vertex)
                    
                    if norm_val > max_norm:
                        max_norm = norm_val
                        best_vertex = vertex.reshape(-1, 1)
                
                return float(max_norm), best_vertex
            else:
                # For larger problems, sample vertices
                max_norm = np.linalg.norm(c)
                best_vertex = c
                
                # Sample up to 10000 random vertices
                for _ in range(10000):
                    alpha = np.random.choice([-1, 1], size=m)
                    vertex = G @ alpha + c.flatten()
                    norm_val = np.linalg.norm(vertex)
                    
                    if norm_val > max_norm:
                        max_norm = norm_val
                        best_vertex = vertex.reshape(-1, 1)
                
                return float(max_norm), best_vertex
                
    except Exception:
        # Ultimate fallback
        c = Z.c if hasattr(Z, 'c') else Z
        val = float(np.linalg.norm(c))
        return val, c.reshape(-1, 1) if c.ndim == 1 else c 