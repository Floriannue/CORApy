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

from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .zonotope import Zonotope

def zonotopeNorm(Z: Zonotope, p: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the norm of the point p w.r.t. the zonotope-norm
    induced by the zonotope Z (see [1, Definition 4]).

    Args:
        Z (Zonotope): A zonotope object.
        p (np.ndarray): A nx1-array, with n the dimension of Z.

    Returns:
        tuple: A tuple containing:
            - res (float): The zonotope-norm of the point p.
            - minimizer (np.ndarray): A solution x s.t. Gx = p and for
              which norm(x,inf) = zonotopeNorm(Z,p).

    References:
        [1] A. Kulmburg, M. Althoff. "On the co-NP-Completeness of the
            Zonotope Containment Problem", European Journal of Control 2021
    """

    # from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
    # equal_dim_check(Z, p)

    if Z.is_empty():
        if p.size == 0:
            return 0, np.array([])
        else:
            return np.inf, np.array([])

    if Z.G is None or Z.G.shape[1] == 0:
        if not np.any(p):
            return 0, np.array([])
        else:
            return np.inf, np.array([])

    n, num_gen = Z.G.shape

    # Set up the linear program as defined in [1, Equation (8)]
    # min t
    # s.t. G*beta = p
    #      -t <= beta_i <= t  for all i
    
    # Objective function: [t, beta_1, ..., beta_m]
    c = np.zeros(num_gen + 1)
    c[0] = 1

    # Equality constraints: G*beta = p  ->  -G*beta + p = 0
    # Rearranged for linprog: [0, G] * [t, beta]^T = p
    A_eq = np.hstack([np.zeros((n, 1)), Z.G])
    b_eq = p.flatten()

    # Inequality constraints: 
    # beta_i - t <= 0  -> [-1, 0, ..., 1, ..., 0] * [t, beta]^T <= 0
    # -beta_i - t <= 0 -> [-1, 0, ...,-1, ..., 0] * [t, beta]^T <= 0
    A_ub_1 = np.hstack([-np.ones((num_gen, 1)), np.eye(num_gen)])
    A_ub_2 = np.hstack([-np.ones((num_gen, 1)), -np.eye(num_gen)])
    A_ub = np.vstack([A_ub_1, A_ub_2])
    b_ub = np.zeros(2 * num_gen)

    # Bounds for t (>=0) and beta (unbounded, handled by inequalities)
    bounds = [(0, None)] + [(None, None)] * num_gen

    # Solve the linear program
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res_lp.success:
        # Infeasible or unbounded
        return np.inf, np.array([])

    res = res_lp.fun
    minimizer = res_lp.x[1:]

    return res, minimizer 