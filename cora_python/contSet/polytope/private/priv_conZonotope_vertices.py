"""
priv_conZonotope_vertices - convert polytope (A,b,Ae,be,V) to conZonotope using vertices

Exact translation of MATLAB logic:
- nrIneq = numel(b); nrEq = numel(be)
- A_all = [A; Ae]; b_all = [b; be]
- minV = min(V,[],2); maxV = max(V,[],2)
- c_ = 0.5*(maxV+minV); G = diag(0.5*(maxV-minV))
- sigma = min(A_all*V,[],2)
- G_ = [G, zeros(n, nrIneq+nrEq)]
- A_ = [A_all*G, diag((sigma-b_all)./2)]
- b_ = (b_all+sigma)./2 - A_all*c_
"""

import numpy as np
from typing import Tuple


def priv_conZonotope_vertices(A: np.ndarray, b: np.ndarray,
                              Ae: np.ndarray, be: np.ndarray,
                              V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = V.shape[0]

    nrIneq = int(b.size)
    nrEq = int(be.size)

    A_all = np.vstack([A, Ae]) if (A.size > 0 or Ae.size > 0) else np.zeros((0, n))
    b_all = np.vstack([b.reshape(-1, 1), be.reshape(-1, 1)]) if (b.size > 0 or be.size > 0) else np.zeros((0, 1))

    minV = np.min(V, axis=1, keepdims=True)
    maxV = np.max(V, axis=1, keepdims=True)

    c_ = 0.5 * (maxV + minV)
    G = np.diagflat(0.5 * (maxV - minV))

    if A_all.shape[0] > 0:
        sigma = np.min(A_all @ V, axis=1, keepdims=True)
    else:
        sigma = np.zeros((0, 1))

    G_ = np.hstack([G, np.zeros((n, nrIneq + nrEq))]) if (nrIneq + nrEq) > 0 else G
    A_left = A_all @ G if G.size > 0 else np.zeros((A_all.shape[0], 0))
    A_right = np.diagflat(((sigma - b_all) / 2.0).flatten()) if (nrIneq + nrEq) > 0 else np.zeros((A_all.shape[0], 0))
    A_ = np.hstack([A_left, A_right]) if (A_right.size + A_left.size) > 0 else np.zeros((A_all.shape[0], G.shape[1]))
    b_ = ((b_all + sigma) / 2.0) - (A_all @ c_)

    return c_, G_, A_, b_


