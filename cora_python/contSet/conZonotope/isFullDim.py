"""
isFullDim - checks if the dimension of the affine hull of a constrained
   zonotope is equal to the dimension of its ambient space

Syntax:
    res = isFullDim(cZ)
    res = isFullDim(cZ, tol)
    [res, subspace] = isFullDim(cZ)

Inputs:
    cZ - conZonotope object
    tol - tolerance

Outputs:
    res - true/false
    subspace - (optional) Returns a set of orthogonal unit vectors
              x_1,...,x_k such that cZ is strictly contained in
              center(cZ)+span(x_1,...,x_k)
              (here, 'strictly' means that k is minimal).
              Note that if cZ is just a point, subspace=[].

Example:
    Z = [0 1.5 -1.5 0.5;0 1 0.5 -1];
    A = [1 1 1]; b = 1;
    cZ1 = conZonotope(Z,A,b);

    P = polytope([],[],[1,-2],1);
    cZ2 = cZ1 & P;

    isFullDim(cZ1)
    isFullDim(cZ2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/isFullDim

Authors:       Niklas Kochdumper, Adrian Kulmburg (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       02-January-2020 (MATLAB)
Last update:   04-February-2025 (AK, implemented subspace computation, MATLAB)
               13-February-2025 (TL, added tol, MATLAB)
Last revision: ---
"""

import numpy as np
import scipy.linalg
from typing import Tuple, Optional


def isFullDim(cZ, tol: float = 1e-8) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Checks if the dimension of the affine hull of a constrained zonotope is
    equal to the dimension of its ambient space.
    """
    from cora_python.contSet.zonotope import Zonotope

    # empty object -> not full-dimensional
    if cZ.representsa_('emptySet', np.finfo(float).eps):
        return False, np.array([])

    if cZ.A.size == 0:
        Z = Zonotope(cZ.c, cZ.G)
        return Z.isFullDim(tol)

    # compute null-space of the constraints
    T = scipy.linalg.null_space(cZ.A)

    # transform generator matrix into the null-space
    G_ = cZ.G @ T

    # check if rank of generator matrix is equal to the dimension
    dimG = G_.shape[0]
    if G_.size == 0:
        U = G_
        r = 0
    else:
        U, Sigma, _ = np.linalg.svd(G_)
        r = np.sum(Sigma > tol)

    res = dimG == r

    if not res:
        subspace = U[:, :r] if G_.size != 0 else np.array([])
    else:
        subspace = np.eye(dimG)

    return res, subspace
