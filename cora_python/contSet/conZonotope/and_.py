"""
and_ - computes the intersection of a constrained zonotope with
   other set representations

Syntax:
    res = and_(cZ,S)

Inputs:
    cZ - conZonotope object
    S - contSet object

Outputs:
    res - conZonotope object

Example:
    % constrained zonotopes
    Z = [0 3 0 1;0 0 2 1];
    A = [1 0 1]; b = 1;
    cZ1 = conZonotope(Z,A,b);
    Z = [0 1.5 -1.5 0.5;0 1 0.5 -1];
    A = [1 1 1]; b = 1;
    cZ2 = conZonotope(Z,A,b);

    % compute intersection
    res1 = cZ1 & cZ2;

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/and

References:
  [1] J. Scott et al. "Constrained zonotope: A new tool for set-based
      estimation and fault detection"

Authors:       Dmitry Grebenyuk, Niklas Kochdumper (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       13-May-2018 (MATLAB)
Last update:   05-May-2020 (MW, standardized error message, MATLAB)
Last revision: 27-March-2023 (MW, rename and_, MATLAB)
               28-September-2024 (MW, integrate precedence, MATLAB)
"""

import numpy as np
from typing import Any

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.conZonotope.compact_ import compact_


def and_(cZ, S: Any, *args):
    """
    Compute the intersection of a constrained zonotope with another set.
    """
    # call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(cZ, 'precedence') and S.precedence < cZ.precedence:
        return S.and_(cZ, *args)

    cls_name = S.__class__.__name__ if hasattr(S, '__class__') else ''

    # constrained zonotope
    if cls_name == 'ConZonotope':
        return _aux_and_conZonotope(cZ, S)

    # higher precedence cases: convert to constrained zonotope
    if cls_name in ('Zonotope', 'Interval', 'ZonoBundle'):
        from cora_python.contSet.conZonotope.conZonotope import ConZonotope
        return _aux_and_conZonotope(cZ, ConZonotope(S))

    raise CORAerror('CORA:noops', cZ, S)


def _aux_and_conZonotope(cZ, S):
    """
    Calculate intersection according to equation (13) at Proposition 1 in [1].
    """
    # MATLAB: Z = [cZ.c, cZ.G, zeros(size(S.G))];
    Z = np.hstack([cZ.c, cZ.G, np.zeros_like(S.G)])

    # MATLAB: A = blkdiag(cZ.A,S.A);
    A = _aux_blkdiag(cZ.A, S.A)
    g1 = cZ.G.shape[1]
    g2 = S.G.shape[1]
    total_cols = g1 + g2
    if cZ.A.size == 0 and S.A.size == 0:
        A = np.zeros((0, total_cols))
    elif cZ.A.size == 0 and S.A.size != 0:
        A = np.hstack([np.zeros((S.A.shape[0], g1)), A])
    elif cZ.A.size != 0 and S.A.size == 0:
        A = np.hstack([A, np.zeros((A.shape[0], g2))])

    # MATLAB: A = [A; cZ.G, -S.G];
    A = np.vstack([A, np.hstack([cZ.G, -S.G])])

    # MATLAB: b = [cZ.b; S.b; S.c - cZ.c];
    b = np.vstack([
        _aux_col(cZ.b),
        _aux_col(S.b),
        _aux_col(S.c - cZ.c)
    ])

    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    res = ConZonotope(Z, A, b)

    # MATLAB: res = compact_(res,'zeros',eps);
    return compact_(res, 'zeros', np.finfo(float).eps)


def _aux_blkdiag(A, B):
    """
    Block diagonal matrix for two inputs.
    """
    if A.size == 0 and B.size == 0:
        return np.zeros((0, 0))
    if A.size == 0:
        return np.asarray(B)
    if B.size == 0:
        return np.asarray(A)
    top = np.hstack([A, np.zeros((A.shape[0], B.shape[1]))])
    bottom = np.hstack([np.zeros((B.shape[0], A.shape[1])), B])
    return np.vstack([top, bottom])


def _aux_col(vec: np.ndarray) -> np.ndarray:
    """Ensure column vector shape for stacking."""
    arr = np.asarray(vec)
    if arr.size == 0:
        return arr.reshape(0, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.shape[1] != 1:
        return arr.reshape(-1, 1)
    return arr
