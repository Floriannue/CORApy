"""
priv_plusEllipsoidOA_halder - Halder (2018) parameterized MVEE of Minkowski sum (outer)
Exact translation of MATLAB priv_plusEllipsoidOA_halder.m
"""

from __future__ import annotations

import numpy as np
from typing import List

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def priv_plusEllipsoidOA_halder(E_cell: List[Ellipsoid]) -> Ellipsoid:
    N = len(E_cell)
    if N == 1:
        return E_cell[0]

    # Initialize with first ellipsoid
    Q = E_cell[0].Q
    q = E_cell[0].q
    TOL = E_cell[0].TOL

    # Accumulate pairwise using Halder's beta iteration
    for i in range(1, N):
        Q1 = Q
        Q2 = E_cell[i].Q
        # eigenvalues of inv(Q1)*Q2
        try:
            vals = np.linalg.eigvals(np.linalg.solve(Q1, Q2))
        except np.linalg.LinAlgError:
            # If Q1 is singular (degenerate), bloat minimally
            U, s, Vh = np.linalg.svd(Q1)
            s = np.array(s)
            s[s < TOL] = TOL
            Q1 = U @ np.diag(s) @ U.T
            vals = np.linalg.eigvals(np.linalg.solve(Q1, Q2))
        l_i = np.real(vals)

        beta = 0.5
        while True:
            denom = (1 + beta * l_i)
            # Protect against numerical issues
            denom = np.where(np.abs(denom) < TOL, np.sign(denom) * TOL, denom)
            num_sum = np.sum(1.0 / denom)
            den_sum = np.sum(l_i / denom)
            if den_sum == 0:
                beta_new = beta
            else:
                beta_new = np.sqrt(num_sum / den_sum)
            if withinTol(beta, beta_new, TOL):
                break
            beta = beta_new
        if beta_new < 0 or withinTol(beta_new, 0, TOL):
            raise CORAerror('CORA:specialError', 'Value almost zero!')

        # Update Q and q
        Q = (1 + 1 / beta_new) * Q1 + (1 + beta_new) * Q2
        q = q + E_cell[i].q

    return Ellipsoid(Q, q)

