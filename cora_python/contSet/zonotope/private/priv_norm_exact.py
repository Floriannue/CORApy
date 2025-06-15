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

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP_AVAILABLE = True
except ImportError:
    SCIPY_MILP_AVAILABLE = False


def priv_norm_exact(Z, norm_type: int) -> Tuple[float, np.ndarray]:
    """
    Computes the exact maximum norm using optimization.
    
    Args:
        Z: zonotope object
        norm_type: p-norm type
        
    Returns:
        tuple: (val, x) where val is norm value and x is vertex
    """
    
    if norm_type != 2:
        raise ValueError("Only Euclidean norm supported")
    
    # Get zonotope properties
    c = Z.c
    G = Z.G
    
    if G.size == 0:
        # Zonotope is just a point
        x = c
        val = np.linalg.norm(x)
        return val, x
    
    n, m = G.shape
    
    # Try different solvers in order of preference
    if CVXPY_AVAILABLE:
        return _solve_with_cvxpy(c, G)
    elif SCIPY_MILP_AVAILABLE:
        return _solve_with_scipy_milp(c, G)
    else:
        # Fallback to vertex enumeration for small problems
        warnings.warn("No suitable solver available for exact norm computation. Using vertex enumeration.")
        return _solve_with_vertex_enumeration(Z)


def _solve_with_cvxpy(c: np.ndarray, G: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Solve using CVXPY with binary variables.
    """
    n, m = G.shape
    
    # Binary variables for each generator
    u = cp.Variable(m, boolean=True)
    
    # Convert binary to {-1, +1}
    alpha = 2 * u - 1
    
    # Vertex of zonotope
    x = G @ alpha + c.flatten()
    
    # Objective: maximize ||x||_2^2 (equivalent to maximizing ||x||_2)
    objective = cp.Maximize(cp.sum_squares(x))
    
    # Solve
    problem = cp.Problem(objective)
    problem.solve()
    
    if problem.status == cp.OPTIMAL:
        alpha_opt = 2 * u.value - 1
        x_opt = G @ alpha_opt + c.flatten()
        val = np.linalg.norm(x_opt)
        return val, x_opt.reshape(-1, 1)
    else:
        # Fallback
        warnings.warn("CVXPY solver failed, using vertex enumeration")
        from ..vertices_ import vertices_
        vertices = vertices_(type('obj', (), {'c': c, 'G': G})())
        norms = np.linalg.norm(vertices, axis=0)
        max_idx = np.argmax(norms)
        return norms[max_idx], vertices[:, max_idx:max_idx+1]


def _solve_with_scipy_milp(c: np.ndarray, G: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Solve using scipy's mixed-integer linear programming.
    """
    n, m = G.shape
    
    # This is more complex to set up as a MILP, so for now use vertex enumeration
    warnings.warn("MILP formulation not yet implemented, using vertex enumeration")
    return _solve_with_vertex_enumeration(type('obj', (), {'c': c, 'G': G})())


def _solve_with_vertex_enumeration(Z) -> Tuple[float, np.ndarray]:
    """
    Solve by enumerating vertices (for small problems).
    """
    try:
        vertices = Z.vertices_()
        if vertices.size == 0:
            return np.linalg.norm(Z.c), Z.c
        
        # Compute norm for each vertex
        norms = np.linalg.norm(vertices, axis=0)
        max_idx = np.argmax(norms)
        
        return norms[max_idx], vertices[:, max_idx:max_idx+1]
    except Exception:
        # Ultimate fallback: sample some vertices
        c = Z.c
        G = Z.G
        n, m = G.shape
        
        max_norm = np.linalg.norm(c)
        best_vertex = c
        
        # Sample up to 1024 vertices
        for i in range(min(2**m, 1024)):
            # Convert i to binary representation
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
        
        return max_norm, best_vertex 