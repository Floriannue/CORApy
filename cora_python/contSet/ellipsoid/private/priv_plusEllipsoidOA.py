"""
priv_plusEllipsoidOA - computes the smallest-volume outer-approximation of the Minkowski sum of ellipsoids
CVXPY implementation adapted from MATLAB formulation.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_plusEllipsoidOA(E_cell: List[Ellipsoid]) -> Ellipsoid:
    N = len(E_cell)
    if N == 1:
        return E_cell[0]

    n = E_cell[0].dim()

    # Bloat degenerate ellipsoids slightly to avoid singular inverses
    max_tol = max(Ei.TOL for Ei in E_cell)
    for i in range(N):
        if not E_cell[i].isFullDim():
            U, s, Vh = np.linalg.svd(E_cell[i].Q)
            nt = E_cell[i].rank()
            s = np.array(s)
            if nt < n:
                fill = 2 * max(s[0] * max_tol, max_tol)
                s[nt:] = fill
            Q_i = U @ np.diag(s) @ U.T
            E_cell[i] = Ellipsoid(0.5 * (Q_i + Q_i.T), E_cell[i].q)

    # Build E0 and per-ellipsoid matrices At_i, bt_i, c_i
    E_blocks = []
    E0 = np.zeros((n, n * N))
    At_list = []
    bt_list = []
    c_list = []
    for i in range(N):
        E_i = np.zeros((n, n * N))
        E_i[:, i * n : (i + 1) * n] = np.eye(n)
        E_blocks.append(E_i)
        E0 = E0 + E_i
        Qinv = np.linalg.pinv(E_cell[i].Q)
        b_i = -Qinv @ E_cell[i].q
        At_list.append(E_i.T @ Qinv @ E_i)
        bt_list.append(E_i.T @ b_i)
        # Fix the deprecation warning by extracting scalar value properly
        c_val = E_cell[i].q.T @ Qinv @ E_cell[i].q - 1
        c_list.append(float(c_val.item() if hasattr(c_val, 'item') else c_val))

    # Variables: symmetric B (n x n), vector b (n x 1), l (N)
    B = cp.Variable((n, n), symmetric=True)
    b = cp.Variable((n, 1))
    l = cp.Variable(N, nonneg=True)

    # Build the constraint matrix C exactly as in MATLAB YALMIP implementation
    # C = [E0'*B*E0, E0'*b, zeros(n*N,n);
    #      b'*E0,    -1,     b';
    #      zeros(n,n*N), b,   -B];
    
    top_left = E0.T @ B @ E0               # (nN x nN)
    top_right = E0.T @ b                   # (nN x 1)
    top_zeros = cp.Constant(np.zeros((n*N, n)))  # (nN x n)
    
    middle_left = (E0.T @ b).T             # (1 x nN)
    middle_middle = cp.Constant([[-1]])    # (1 x 1)
    middle_right = b.T                     # (1 x n)
    
    bottom_left = cp.Constant(np.zeros((n, n*N)))  # (n x nN)
    bottom_middle = b                      # (n x 1)
    bottom_right = -B                      # (n x n)
    
    C = cp.bmat([[top_left, top_right, top_zeros],
                  [middle_left, middle_middle, middle_right],
                  [bottom_left, bottom_middle, bottom_right]])

    # For each ellipsoid i: subtract l_i * [At_c{i}, bt_c{i}, zeros(n*N,n);
    #                                        bt_c{i}', c_c{i}, zeros(1,n);
    #                                        zeros(n,n*N+1+n)]
    for i in range(N):
        At_i = At_list[i]                  # (nN x nN)
        bt_i = bt_list[i]                  # (nN x 1)
        c_i = c_list[i]                    # scalar
        
        # Build the matrix to subtract
        sub_top_left = At_i                 # (nN x nN)
        sub_top_right = bt_i                # (nN x 1)
        sub_top_zeros = cp.Constant(np.zeros((n*N, n)))  # (nN x n)
        
        sub_middle_left = bt_i.T            # (1 x nN)
        sub_middle_middle = cp.Constant([[c_i]])  # (1 x 1)
        sub_middle_right = cp.Constant(np.zeros((1, n)))  # (1 x n)
        
        sub_bottom_left = cp.Constant(np.zeros((n, n*N)))  # (n x nN)
        sub_bottom_middle = cp.Constant(np.zeros((n, 1)))  # (n x 1)
        sub_bottom_right = cp.Constant(np.zeros((n, n)))   # (n x n)
        
        C_i = cp.bmat([[sub_top_left, sub_top_right, sub_top_zeros],
                       [sub_middle_left, sub_middle_middle, sub_middle_right],
                       [sub_bottom_left, sub_bottom_middle, sub_bottom_right]])
        
        C = C - l[i] * C_i

    # Constraint: C <= 0 (negative semidefinite)
    constraints = [C << 0]

    # Objective: minimize -logdet(B^{-1}) == maximize logdet(B)
    prob = cp.Problem(cp.Maximize(cp.log_det(B)), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise CORAerror('CORA:solverIssue', f'Minkowski sum OA SDP failed: {prob.status}')

    Bsol = B.value
    bsol = b.value
    Q = np.linalg.pinv(Bsol)
    q = -np.linalg.solve(Bsol, bsol)
    return Ellipsoid(Q, q)

