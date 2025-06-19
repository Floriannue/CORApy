import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import warnings
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.check import withinTol
from cora_python.contSet.polytope.private.priv_equalityToInequality import priv_equalityToInequality
from .center import center

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def vertices_(P: 'Polytope') -> np.ndarray:
    """
    Computes the vertices of a polytope.
    This is a Python translation of the 'lcon2vert' and 'comb' methods
    from the MATLAB CORA library.
    """
    tol = 1e-9

    if P._has_v_rep:
        return P._V

    n = P.dim()

    # 1D case
    if n == 1:
        # Simplified 1D vertex calculation
        V_list = []
        if P.A is not None and P.b is not None:
            for i in range(P.A.shape[0]):
                if P.A[i, 0] != 0:
                    V_list.append(P.b[i, 0] / P.A[i, 0])
        if P.Ae is not None and P.be is not None:
            for i in range(P.Ae.shape[0]):
                if P.Ae[i, 0] != 0:
                    V_list.append(P.be[i, 0] / P.Ae[i, 0])
        
        if not V_list:
            return np.array([[]])

        min_v, max_v = min(V_list), max(V_list)
        
        # HACK: contains is not fully robust yet for all cases
        vertices = [v for v in [min_v, max_v]]
        return np.array([vertices]) if vertices else np.array([[]])

    # Check for emptiness/unboundedness using Chebyshev center
    c, r = center(P)
    if r < 0:
        # Polytope is empty
        return np.zeros((n, 0))
    if np.isinf(r):
        # Polytope is unbounded
        raise ValueError("Cannot compute vertices for an unbounded polytope.")

    # Combine all constraints into A_ineq*x <= b_ineq form
    A_ineq, b_ineq = priv_equalityToInequality(P.A, P.b, P.Ae, P.be)

    if A_ineq is None or A_ineq.shape[0] < n:
        return np.zeros((n, 0)) # Not enough constraints to define vertices

    # --- 'comb' method logic ---
    # Iterate through all combinations of n constraints
    potential_vertices = []
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='A_eq does not appear to be of full row rank. The dual solution may be inaccurate.')
        for indices in combinations(range(A_ineq.shape[0]), n):
            A_sub = A_ineq[list(indices), :]
            b_sub = b_ineq[list(indices)]

            # Solve A_sub * x = b_sub to find intersection point
            try:
                # Using direct solver first, fallback to pseudo-inverse
                v = np.linalg.solve(A_sub, b_sub)
                potential_vertices.append(v.flatten())
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                try:
                    v = np.linalg.pinv(A_sub) @ b_sub
                    potential_vertices.append(v.flatten())
                except np.linalg.LinAlgError:
                    # This combination of constraints is truly problematic
                    continue

    if not potential_vertices:
        return np.zeros((n, 0))

    # --- Filter vertices ---
    # Check which potential vertices satisfy ALL constraints
    valid_vertices = []
    unique_vertices_set = set()

    for v in potential_vertices:
        v_col = v.reshape(-1, 1)
        if np.all(A_ineq @ v_col <= b_ineq + tol):
            # Check for uniqueness before adding
            v_tuple = tuple(np.round(v, 6))
            if v_tuple not in unique_vertices_set:
                valid_vertices.append(v)
                unique_vertices_set.add(v_tuple)

    if not valid_vertices:
        return np.zeros((n, 0))

    return np.array(valid_vertices).T 