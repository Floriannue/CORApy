"""
randPoint_ - generates random points within a polytope (MATLAB parity style)

Supported types:
- 'standard': convex combination of vertices (uniform over simplex)
- 'extreme': pick extreme points (vertices) along random directions
- for fullspace: Gaussian sampling

Notes:
- If only H-rep available, vertices_() is computed and used.
- Empty set returns (n x 0) points.
"""

import numpy as np
from typing import Union, TYPE_CHECKING

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def randPoint_(P: 'Polytope', N: Union[int, str] = 1, type_: str = 'standard') -> np.ndarray:
    n = P.dim()

    # Fullspace -> Gaussian (detect directly via empty H-representation)
    P.constraints()
    if P.A.size == 0 and P.Ae.size == 0:
        if isinstance(N, str):
            raise CORAerror('CORA:wrongInput', 'N must be integer for fullspace sampling')
        return np.random.randn(n, int(N))

    # Empty -> no points
    if P.representsa_('emptySet', 1e-12):
        return np.zeros((n, 0))

    # Treat explicit empty objects fast-path (after fullspace/empty checks)
    if hasattr(P, 'isemptyobject') and P.isemptyobject():
        return np.zeros((n, 0))

    if isinstance(N, str) and N == 'all':
        # For polytopes, 'all' for extreme returns vertices
        V = P.vertices_()
        return V

    if type_ not in ('standard', 'extreme'):
        raise CORAerror('CORA:noSpecificAlg', f'{type_} not supported for polytope')

    N = int(N)

    # Ensure we have vertices
    V = P.V if P.isVRep and getattr(P, '_V', None) is not None and P._V.size > 0 else P.vertices_()
    if V.size == 0:
        return np.zeros((n, 0))

    if type_ == 'standard':
        # Random convex combinations using Dirichlet sampling
        m = V.shape[1]
        if m == 1:
            return np.repeat(V, N, axis=1)
        alpha = np.random.rand(m, N)
        alpha /= np.sum(alpha, axis=0, keepdims=True)
        pts = V @ alpha
        return pts

    # type_ == 'extreme'
    m = V.shape[1]
    if m == 1:
        return np.repeat(V, N, axis=1)
    pts = np.zeros((n, N))
    # Center for stability
    c = P.center().reshape(-1, 1)
    Vc = V - c
    for i in range(N):
        d = np.random.randn(n, 1)
        d /= np.linalg.norm(d)
        idx = int(np.argmax((d.T @ Vc).flatten()))
        pts[:, i:i+1] = V[:, [idx]]
    return pts


