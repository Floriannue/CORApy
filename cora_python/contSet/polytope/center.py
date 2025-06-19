"""
center - center of a polytope

Syntax:
    c = center(P)

Inputs:
    P - polytope object

Outputs:
    c - center of the polytope

Example:
    A = [1 0; 0 1; -1 0; 0 -1];
    b = [1; 1; 1; 1];
    P = polytope(A, b);
    c = center(P)

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 16-September-2019 (MW, specify output for empty case) (MATLAB)
Python translation: 2025
"""

import numpy as np
from scipy.optimize import linprog
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .polytope import Polytope

def center(P: 'Polytope') -> Tuple[np.ndarray, float]:
    """
    Computes the Chebyshev center of a polytope.
    The Chebyshev center is the center of the largest inscribed sphere in the
    polytope. It is found by solving a linear program.

    Args:
        P: A Polytope object.

    Returns:
        A tuple (c, r) where c is the center (n x 1) and r is the radius.
        If the polytope is empty, c is an empty array and r is -1.
        If the polytope is unbounded, c is an array of NaNs and r is inf.
    """
    n = P.dim()

    if n == 0:
        return np.array([]), -1 # Empty

    # The H-representation is required
    A = P.A
    b = P.b
    Ae = P.Ae
    be = P.be
    
    # Setup the linear program for the Chebyshev center
    # The variables are [c_1, ..., c_n, r]
    # We want to maximize r, which is equivalent to minimizing -r.
    # So, the cost function is [0, ..., 0, -1]
    f = np.zeros(n + 1)
    f[-1] = -1

    # The constraints are:
    # A_i * c + r * ||A_i|| <= b_i  for all inequality constraints
    # Ae_j * c = be_j               for all equality constraints
    # r >= 0

    # Inequality constraints
    A_ineq = None
    b_ineq = None
    if A is not None and A.size > 0:
        A_norm = np.linalg.norm(A, axis=1, keepdims=True)
        A_ineq = np.hstack([A, A_norm])
        b_ineq = b

    # Equality constraints
    A_eq = None
    b_eq = None
    if Ae is not None and Ae.size > 0:
        A_eq = np.hstack([Ae, np.zeros((Ae.shape[0], 1))])
        b_eq = be

    # Bounds for the variables
    # c can be anything, r must be non-negative
    bounds = [(None, None)] * n + [(0, None)]

    # Solve the LP
    # We are minimizing -r, so the result for r will be -res.fun
    res = linprog(c=f, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')

    if res.success:
        c = res.x[:n].reshape(-1, 1)
        r = res.x[n]
        return c, r
    elif res.status == 2: # Infeasible
        return np.array([]), -1
    elif res.status == 3: # Unbounded
        return np.full((n, 1), np.nan), np.inf
    else:
        # Some other solver issue
        # This might happen for unbounded cases where the solver
        # doesn't explicitly return status 3.
        # We'll treat it as unbounded as a fallback.
        return np.full((n, 1), np.nan), np.inf


def _is_box_polytope(P) -> bool:
    """Check if polytope represents a box (axis-aligned)"""
    if P.A is None or P.b is None:
        return False
    
    # Check if A matrix has only 0, 1, -1 entries and each row has exactly one non-zero
    A_abs = np.abs(P.A)
    row_sums = np.sum(A_abs, axis=1)
    
    # Each row should have exactly one non-zero element
    if not np.allclose(row_sums, 1.0):
        return False
    
    # Non-zero elements should be 1 or -1
    non_zero_mask = A_abs > 1e-12
    if not np.allclose(A_abs[non_zero_mask], 1.0):
        return False
    
    return True


def _compute_box_center(P) -> np.ndarray:
    """Compute center for box polytope"""
    n = P.A.shape[1]
    center = np.zeros((n, 1))
    
    for i in range(n):
        # Find positive and negative constraints for dimension i
        pos_mask = (P.A[:, i] > 0.5)
        neg_mask = (P.A[:, i] < -0.5)
        
        if np.any(pos_mask) and np.any(neg_mask):
            # Compute bounds
            upper_bound = np.min(P.b[pos_mask])
            lower_bound = -np.max(P.b[neg_mask])
            center[i] = (upper_bound + lower_bound) / 2
    
    return center 