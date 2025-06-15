"""
zonotopeNorm - computes the norm of the point p w.r.t. the zonotope-norm
    induced by the zonotope Z (see [1, Definition 4]).

Syntax:
    res = zonotopeNorm(Z, p)
    res, minimizer = zonotopeNorm(Z, p)

Inputs:
    Z - zonotope
    p - nx1-array, with n the dimension of Z

Outputs:
    res - zonotope-norm of the point p
    minimizer - (optional) returns a solution x s.t. Gx = p and for
                 which norm(x,inf) = zonotopeNorm(Z,p)

Example:
    c = [0;0]
    G = [[2 3 0];[2 0 3]]
    Z = zonotope(c, G)
    
    p = rand([2 1])

    # Set of points that have the same distance to the origin as p, with
    # respect to the zonotope norm of Z    
    d = zonotopeNorm(Z, p)

References:
    [1] A. Kulmburg, M. Althoff. "On the co-NP-Completeness of the
        Zonotope Containment Problem", European Journal of Control 2021

Authors:       Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       14-May-2021 (MATLAB)
Last update:   16-January-2024 (MW, handle edge cases)
Last revision: ---
"""

import numpy as np
from typing import Tuple, Optional, Union

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def zonotopeNorm(Z, p: np.ndarray, return_minimizer: bool = False) -> Union[float, Tuple[float, Optional[np.ndarray]]]:
    """
    Computes the norm of the point p w.r.t. the zonotope-norm induced by the zonotope Z.
    
    Args:
        Z: zonotope object
        p: point (n x 1 array)
        return_minimizer: if True, also return the minimizer
        
    Returns:
        float or tuple: zonotope norm, optionally with minimizer
    """
    
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for zonotopeNorm computation")
    
    # Ensure p is a column vector
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    elif p.ndim == 2 and p.shape[1] != 1:
        raise ValueError("p must be a column vector")
    
    p = p.flatten()  # For linear programming, we need 1D
    
    # Check if zonotope represents empty set
    from .representsa_ import representsa_
    if representsa_(Z, 'emptySet', 1e-15):
        if p.size == 0:
            res = 0.0
            minimizer = np.array([])
        else:
            res = np.inf
            minimizer = np.array([])
        
        if return_minimizer:
            return res, minimizer
        else:
            return res
    
    # Get generator matrix
    G = Z.G
    if G.size == 0:
        if np.allclose(p, 0):
            res = 0.0
            minimizer = np.array([])
        else:
            res = np.inf
            minimizer = np.array([])
        
        if return_minimizer:
            return res, minimizer
        else:
            return res
    
    n, num_gen = G.shape
    
    # Set up linear program as defined in the paper
    # Variables: [t, x1, x2, ..., x_numGen]
    # Minimize t subject to:
    # G * x = p (equality constraint)
    # -t <= x_i <= t for all i (inequality constraints)
    
    # Objective: minimize t
    c = np.zeros(1 + num_gen)
    c[0] = 1.0
    
    # Equality constraints: G * x = p
    # [0, G] * [t; x] = p
    A_eq = np.hstack([np.zeros((n, 1)), G])
    b_eq = p
    
    # Inequality constraints: -t <= x_i <= t for all i
    # This becomes: x_i - t <= 0 and -x_i - t <= 0
    A_ub = np.zeros((2 * num_gen, 1 + num_gen))
    b_ub = np.zeros(2 * num_gen)
    
    for i in range(num_gen):
        # x_i - t <= 0
        A_ub[2*i, 0] = -1  # -t
        A_ub[2*i, 1+i] = 1  # x_i
        
        # -x_i - t <= 0
        A_ub[2*i+1, 0] = -1  # -t
        A_ub[2*i+1, 1+i] = -1  # -x_i
    
    # Bounds: t >= 0, x_i unbounded (handled by inequality constraints)
    bounds = [(0, None)] + [(None, None)] * num_gen
    
    try:
        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            res = result.x[0]  # The optimal t value
            minimizer = result.x[1:] if return_minimizer else None
        else:
            # Problem is infeasible or unbounded
            res = np.inf
            minimizer = None
            
    except Exception:
        # Solver failed
        res = np.inf
        minimizer = None
    
    if return_minimizer:
        return res, minimizer
    else:
        return res 