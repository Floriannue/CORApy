"""
priv_conZonotope_supportFunc - convert polytope (A,b,Ae,be) to conZonotope using support functions

Exact translation of MATLAB logic:
- c_ = center(B)
- G = diag(0.5 * (supremum(B) - infimum(B)))
- A_all = [A; Ae], b_all = [b; be]
- sigma(a) = supportFunc_(B, A_all(a,:)', 'lower'); if any is Inf -> empty
- G_ = [G, zeros(n, nrIneq+nrEq)]
- A_ = [A_all*G, diag((sigma-b_all)./2)]
- b_ = (b_all+sigma)./2 - A_all*c_
"""

import numpy as np
from typing import Tuple

def priv_conZonotope_supportFunc(A: np.ndarray, b: np.ndarray,
                                 Ae: np.ndarray, be: np.ndarray,
                                 B) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    n = A.shape[1] if A.size > 0 else (Ae.shape[1] if Ae.size > 0 else B.dim())

    c_ = B.center()
    sup = B.supremum()
    inf = B.infimum()
    G = np.diagflat(0.5 * (sup - inf))

    nrIneq = int(b.size)
    nrEq = int(be.size)

    A_all = np.vstack([A, Ae]) if (A.size > 0 or Ae.size > 0) else np.zeros((0, n))
    b_all = np.vstack([b.reshape(-1, 1), be.reshape(-1, 1)]) if (b.size > 0 or be.size > 0) else np.zeros((0, 1))

    # compute lower bound sigma for A*x in [sigma, b]
    sigma_list = []
    for a_idx in range(nrIneq):
        a_row = A_all[a_idx, :].reshape(-1, 1)
        val = B.supportFunc_(a_row, 'lower')
        val_num = val[0] if isinstance(val, tuple) else val
        if np.isinf(val_num):
            return np.array([]), np.array([]), np.array([]), np.array([]), True
        sigma_list.append(val_num)
    # equality constraints: append be
    if nrEq > 0:
        sigma = np.vstack([np.array(sigma_list).reshape(-1, 1), be.reshape(-1, 1)])
    else:
        sigma = np.array(sigma_list).reshape(-1, 1)

    G_ = np.hstack([G, np.zeros((G.shape[0], nrIneq + nrEq))]) if (nrIneq + nrEq) > 0 else G
    A_left = A_all @ G if G.size > 0 else np.zeros((A_all.shape[0], 0))
    A_right = np.diagflat(((sigma - b_all) / 2.0).flatten()) if (nrIneq + nrEq) > 0 else np.zeros((A_all.shape[0], 0))
    A_ = np.hstack([A_left, A_right]) if (A_right.size + A_left.size) > 0 else np.zeros((A_all.shape[0], G.shape[1]))
    b_ = ((b_all + sigma) / 2.0) - (A_all @ c_)

    return c_, G_, A_, b_, False


