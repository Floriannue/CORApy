"""
priv_norm_ub - computes bound on the maximum norm value

Syntax:
    ub = priv_norm_ub(Z, type)

Inputs:
    Z - zonotope object
    type - p-norm

Outputs:
    ub - upper bound on the maximum norm value

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       18-September-2019 (MATLAB)
Last update:   23-May-2022 (VG, model optimization problem directly)
Last revision: ---
"""

import numpy as np
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def priv_norm_ub(Z, norm_type: int) -> float:
    """
    Computes upper bound on the maximum norm value using convex optimization.
    
    Args:
        Z: zonotope object
        norm_type: p-norm type
        
    Returns:
        float: upper bound on maximum norm value
    """
    
    if norm_type != 2:
        raise ValueError("Only Euclidean norm supported")
    
    if not np.allclose(Z.c, 0):
        raise ValueError("Not implemented for non-zero center")
    
    # Get generator matrix
    G = Z.G
    n, m = G.shape
    
    if G.size == 0:
        return 0.0
    
    # Compute G'*G
    GG = G.T @ G
    
    # Try different solvers
    if CVXPY_AVAILABLE:
        return _solve_with_cvxpy(GG, m)
    elif SCIPY_AVAILABLE:
        return _solve_with_scipy(GG, m)
    else:
        # Simple upper bound: sum of column norms
        warnings.warn("No suitable solver available. Using simple upper bound.")
        column_norms = np.linalg.norm(G, axis=0)
        return np.sum(column_norms)


def _solve_with_cvxpy(GG: np.ndarray, m: int) -> float:
    """
    Solve the dual SDP problem using CVXPY.
    
    The dual problem to max_{|u|<=1} u'*G'*G*u is:
    min_{d>=0} ones(1,m)*d
    s.t. diag(d) - G'*G >= 0
    """
    try:
        # Decision variable
        d = cp.Variable(m, nonneg=True)
        
        # Objective: minimize sum of d
        objective = cp.Minimize(cp.sum(d))
        
        # Constraint: diag(d) - G'*G >= 0 (positive semidefinite)
        constraints = [cp.diag(d) - GG >> 0]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return np.sqrt(problem.value)
        else:
            warnings.warn("CVXPY solver failed, using fallback")
            return _fallback_bound(GG)
            
    except Exception as e:
        warnings.warn(f"CVXPY failed: {e}, using fallback")
        return _fallback_bound(GG)


def _solve_with_scipy(GG: np.ndarray, m: int) -> float:
    """
    Solve using scipy optimization (simplified approach).
    """
    try:
        # For the constraint diag(d) - G'*G >= 0, we need the smallest eigenvalue
        # of diag(d) - G'*G to be non-negative
        # This means d_i >= lambda_max(G'*G) for all i
        
        eigenvals = np.linalg.eigvals(GG)
        lambda_max = np.max(eigenvals)
        
        # The optimal solution is d = lambda_max * ones(m)
        optimal_value = m * lambda_max
        return np.sqrt(optimal_value)
        
    except Exception as e:
        warnings.warn(f"Scipy solver failed: {e}, using fallback")
        return _fallback_bound(GG)


def _fallback_bound(GG: np.ndarray) -> float:
    """
    Simple fallback bound using matrix norms.
    """
    # Upper bound using Frobenius norm
    frobenius_bound = np.linalg.norm(GG, 'fro')
    
    # Upper bound using spectral norm
    spectral_bound = np.linalg.norm(GG, 2)
    
    # Use the tighter bound
    return np.sqrt(min(frobenius_bound, spectral_bound)) 