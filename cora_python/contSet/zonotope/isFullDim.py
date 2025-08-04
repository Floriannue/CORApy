"""
isFullDim - checks if the dimension of the affine hull of a zonotope is
    equal to the dimension of its ambient space

Syntax:
    res = isFullDim(Z)
    res = isFullDim(Z, tol)
    [res, subspace] = isFullDim(_)

Inputs:
    Z - zonotope object
    tol - numeric, tolerance

Outputs:
    res - true/false
    subspace - (optional) Returns a set of orthogonal unit vectors
               x_1,...,x_k such that Z is strictly contained in
               center(Z)+span(x_1,...,x_k)
               (here, 'strictly' means that k is minimal).
               Note that if Z is just a point, subspace=[].

Example: 
    Z1 = Zonotope(np.array([[1, 2, 1], [3, 1, 2]]))
    Z2 = Zonotope(np.array([[1, 2, 1], [3, 4, 2]]))

    isFullDim(Z1)
    isFullDim(Z2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: isempty

Authors:       Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       02-January-2020 (MATLAB)
Last update:   17-May-2024 (TL, added tol) (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Tuple, Optional
from .zonotope import Zonotope


def isFullDim(Z: Zonotope, tol: float = 1e-6) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Checks if the dimension of the affine hull of a zonotope is
    equal to the dimension of its ambient space
    
    Args:
        Z: zonotope object
        tol: tolerance for rank computation
        
    Returns:
        res: True if full-dimensional, False otherwise
        subspace: (optional) Returns a set of orthogonal unit vectors
                 x_1,...,x_k such that Z is strictly contained in
                 center(Z)+span(x_1,...,x_k)
    """
    if not Z.representsa_('emptySet', np.finfo(float).eps):
        Zdim = Z.dim()
        
        # Compute SVD of generator matrix
        U, Sigma, _ = np.linalg.svd(Z.G, full_matrices=True)
        
        # Count significant singular values
        s = Sigma
        r = np.sum(s > tol)
        
        res = Zdim == r
        
        if not res:
            # Return subspace spanned by significant singular vectors
            subspace = U[:, :r]
        else:
            # Full-dimensional case
            subspace = np.eye(Zdim)
    else:
        # Empty set case
        res = False
        subspace = None
    
    return res, subspace 